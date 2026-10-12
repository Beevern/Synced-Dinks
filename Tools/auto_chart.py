#!/usr/bin/env python3
"""
Auto-chart a song from its audio + pasted lyrics: one target per sung word.

Pipeline:
  1. Demucs (htdemucs) isolates the vocal stem.
  2. wav2vec2 CTC forced alignment finds roughly when each lyric word starts.
  3. Each word is snapped to the nearest vocal onset (librosa) for tight timing.
  4. Writes a marker CSV the game already reads for custom levels
     (header row, timecode in column K, 30 fps, timeline starting at 01:00:00:00).

Usage:
  python auto_chart.py song.mp3 lyrics.txt
  python auto_chart.py song.mp3 lyrics.txt -o notes.csv --offset -0.02

lyrics.txt is just the lyrics pasted as plain text. Section labels like [Chorus]
and lines starting with # are ignored. Write every repeat out in full ("x2" doesn't work) and spell numbers
as words, since the aligner can only place words that are in the text.

The vocal stem is cached as <song>_vocals.wav, so re-running after fixing the
lyrics skips the slow separation step.
"""

import argparse
import csv
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from pitch import fmt, heights, note_pitch, pitch_track

FPS = 30.0               # timeline frame rate for timecodes, EDL and lyric video; set from --fps / the video
STANDARD_FPS = [Fraction(24000, 1001), 24, 25, Fraction(30000, 1001), 30, 48, 50, Fraction(60000, 1001), 60]
SPLIT_FRACTION = 0.6      # multi-syllable words sung this much of the hold length become taps
TC_START_HOURS = 1       # CsvParser subtracts 1 hour (timeline starts at 01:00:00:00)
TIME_COLUMN = 10         # column K
ALIGN_SR = 16000         # wav2vec2 sample rate
CHUNK_SECONDS = 20.0     # wav2vec2 attention is O(n^2), so run it in chunks
CHUNK_CONTEXT = 1.0      # extra audio each side of a chunk, trimmed afterwards
ALIGN_LAG = 0.08         # CTC tends to emit a word's first letter ~80 ms after it starts


# ---------------------------------------------------------------- separation

STEM_FILES = ("vocals", "drums", "instrumental")   # cached next to the song as <song>_<name>.wav


def separate_stems(song_path: Path, device: str) -> tuple[dict[str, np.ndarray], int]:
    """Demucs htdemucs. Returns stereo vocals, drums, and instrumental (everything but vocals)."""
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.eval()
    sr = model.samplerate

    y, _ = librosa.load(str(song_path), sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    wav = torch.from_numpy(y).float()

    # Same normalisation demucs' own CLI does
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, split=True,
                              overlap=0.25, progress=True)[0]
    sources = (sources * std + mean).cpu().numpy()
    names = list(model.sources)
    stems = {"vocals": sources[names.index("vocals")], "drums": sources[names.index("drums")]}
    stems["instrumental"] = sum(sources[i] for i, n in enumerate(names) if n != "vocals")
    return stems, sr


def load_or_separate(song_path: Path, device: str, force: bool) -> tuple[dict[str, np.ndarray], int]:
    """Stems for the song, cached next to it so re-runs skip the slow separation."""
    paths = {n: song_path.with_name(f"{song_path.stem}_{n}.wav") for n in STEM_FILES}
    if all(p.exists() for p in paths.values()) and not force:
        print(f"Using cached stems: {paths['vocals'].parent}")
        stems = {}
        for n, p in paths.items():
            y, sr = sf.read(str(p), always_2d=True)
            stems[n] = y.T.astype(np.float32)
        return stems, sr

    print("Separating vocals with Demucs (this is the slow part)...")
    stems, sr = separate_stems(song_path, device)
    for n, p in paths.items():
        sf.write(str(p), np.clip(stems[n], -1, 1).T, sr)
    print(f"Saved stems: {paths['vocals'].parent}")
    return stems, sr


def drum_hits(drums: np.ndarray, sr: int, min_strength: float = 0.15) -> list[tuple[float, str, float]]:
    """Every drum hit in the drum stem as (seconds, type, strength 0-1).
    type is kick / snare / hihat: whichever frequency band jumps most at that moment."""
    y = librosa.to_mono(drums)
    hop, n_fft = 256, 1024
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(spec, ref=np.max), sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, backtrack=False)
    bands = {"kick": (20, 150), "snare": (150, 2500), "hihat": (5000, sr / 2)}
    band_env = {}
    for name, (lo, hi) in bands.items():
        band = librosa.amplitude_to_db(spec[(freqs >= lo) & (freqs < hi)], ref=np.max)
        flux = np.maximum(np.diff(band, axis=1, prepend=band[:, :1]), 0).mean(axis=0)
        band_env[name] = flux / (np.percentile(flux, 99) + 1e-9)
    top = np.percentile(env, 99) + 1e-9
    hits = []
    for f in frames:
        strength = float(np.clip(env[f] / top, 0, 1))
        if strength < min_strength:
            continue
        kind = max(bands, key=lambda b: band_env[b][min(f, len(band_env[b]) - 1)])
        hits.append((f * hop / sr, kind, strength))
    return hits


def write_drums_csv(path: Path, hits: list[tuple[float, str, float]]):
    """Drum hits for the game (drums.csv in a level folder): one row per hit."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["#", "Seconds", "Type", "Strength"])
        for i, (t, kind, s) in enumerate(hits, 1):
            w.writerow([i, f"{t:.3f}", kind, f"{s:.2f}"])


# ------------------------------------------------------------------ lyrics

def parse_lyrics(text: str) -> list[dict]:
    """One dict per lyric line that has words:
      words  - normalised words to align
      pieces - (word as written, separator after it, adlib) for each word.
               adlib = inside (parentheses), i.e. backing vocals: still aligned
               so the timing around them is right, but no notes or subtitles."""
    text = re.sub(r"\[[^\]]*\]", " ", text)          # drop [Chorus], [Verse 2] ...
    text = text.replace("’", "'").replace("‘", "'")
    lines = []
    for raw_line in text.splitlines():
        if raw_line.lstrip().startswith("#"):      # notes, e.g. the lyrics link in lyrics.txt
            continue
        words, pieces, depth = [], [], 0
        parts = re.split(r"([\s\-–—/]+)", " ".join(raw_line.split()))  # even = tokens
        for k, part in enumerate(parts):
            if k % 2:
                if pieces:
                    pieces[-1][1] += part
                continue
            adlib = depth > 0 or "(" in part
            depth = max(0, depth + part.count("(") - part.count(")"))
            w = re.sub(r"[^A-Z']", "", part.upper()).strip("'")
            if w:
                words.append(w)
                pieces.append([part, "", adlib])
            elif pieces:
                pieces[-1][1] += part
        if words:
            lines.append({"words": words, "pieces": [tuple(p) for p in pieces]})
    return lines


def clean_word(word: str) -> str:
    """Strip punctuation for display, keeping apostrophes inside words (I'm, don't)."""
    return re.sub(r"[^\w']", "", word).strip("'")


# --------------------------------------------------------------- alignment

def compute_emissions(audio16k: np.ndarray, model_name: str, device: str):
    """Returns (log_probs[T, C], frame_times[T], vocab dict)."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()

    total = len(audio16k) / ALIGN_SR
    emissions, times = [], []
    start = 0.0
    while start < total:
        end = min(start + CHUNK_SECONDS, total)
        a = max(0.0, start - CHUNK_CONTEXT)
        b = min(total, end + CHUNK_CONTEXT)
        chunk = audio16k[int(a * ALIGN_SR):int(b * ALIGN_SR)]

        inputs = processor(chunk, sampling_rate=ALIGN_SR, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(inputs.to(device)).logits[0]
        logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

        frame_dur = (b - a) / logp.shape[0]
        t = a + (np.arange(logp.shape[0]) + 0.5) * frame_dur
        keep = (t >= start) & (t < end)
        emissions.append(logp[keep])
        times.append(t[keep])
        start = end

    vocab = processor.tokenizer.get_vocab()
    return np.concatenate(emissions), np.concatenate(times), vocab


def force_align(logp: np.ndarray, tokens: np.ndarray, blank: int) -> np.ndarray:
    """Viterbi CTC alignment. Returns the frame index where each token is first emitted."""
    T, J = logp.shape[0], len(tokens)
    if T < J:
        raise ValueError(f"Lyrics are too long for the audio ({J} tokens, {T} frames).")

    score = np.full(J + 1, -np.inf, dtype=np.float32)
    score[0] = 0.0
    moved = np.zeros((T + 1, J + 1), dtype=bool)
    tok_logp = logp[:, tokens]

    for t in range(T):
        stay = score + logp[t, blank]
        adv = np.full(J + 1, -np.inf, dtype=np.float32)
        adv[1:] = score[:-1] + tok_logp[t]
        moved[t + 1] = adv > stay
        score = np.maximum(stay, adv)

    starts = np.zeros(J, dtype=np.int64)
    j = J
    for t in range(T, 0, -1):
        if j > 0 and moved[t, j]:
            starts[j - 1] = t - 1
            j -= 1
    return starts


def align_words(vocals: np.ndarray, sr: int, words: list[str], model_name: str, device: str):
    mono = librosa.to_mono(vocals)
    audio16k = librosa.resample(mono, orig_sr=sr, target_sr=ALIGN_SR)

    print(f"Aligning {len(words)} words with {model_name}...")
    logp, frame_times, vocab = compute_emissions(audio16k, model_name, device)

    blank = vocab.get("<pad>", 0)
    sep = vocab["|"]
    tokens, word_first_tok, word_tok_ranges = [], [], []
    for i, w in enumerate(words):
        if i > 0:
            tokens.append(sep)
        first = len(tokens)
        tokens.extend(vocab[c] for c in w if c in vocab)
        word_first_tok.append(first)
        word_tok_ranges.append((first, len(tokens)))

    starts = force_align(logp, np.array(tokens), blank)

    aligned = []
    for w, (a, b) in zip(words, word_tok_ranges):
        frames = starts[a:b]
        conf = float(np.exp(logp[frames, np.array(tokens[a:b])].mean()))
        t = max(0.0, float(frame_times[starts[a]]) - ALIGN_LAG)
        aligned.append({"word": w, "aligned": t, "confidence": conf})
    return aligned


# ------------------------------------------------------------------ onsets

def vocal_onsets(vocals: np.ndarray, sr: int) -> np.ndarray:
    mono = librosa.to_mono(vocals)
    hop = 256
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    return librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop,
                                      units="time", backtrack=False)


def snap_to_onsets(aligned: list[dict], onsets: np.ndarray, window: float, min_gap: float):
    """Move each word to the nearest unused onset within `window`, keeping order."""
    last = -np.inf
    for item in aligned:
        t = item["aligned"]
        lo = max(t - window, last + min_gap)
        cands = onsets[(onsets >= lo) & (onsets <= t + window)]
        if len(cands):
            item["time"] = float(cands[np.argmin(np.abs(cands - t))])
            item["snapped"] = True
        else:
            item["time"] = t
            item["snapped"] = False
        last = item["time"]
    return aligned


def estimate_durations(items: list[dict], vocals: np.ndarray, sr: int, drop_db: float = 20.0,
                       sustain_drop_db: float = 14.0, gap_frames: int = 3,
                       words: list[float] | None = None):
    """How long each word is sung. Tuned against hand-charted holds:
      1. until the vocal level falls `drop_db` below the word's attack peak, or the next word starts
      2. for long notes, earlier if it falls `sustain_drop_db` below the sustained part of the note
         (echo and backing vocals keep the level up after the singer stops, so rule 1 alone
         runs line-ending notes ~1 s long)
      3. always ending `gap_frames` frames before the next note, so a hold's end marker never
         lands on the next note's frame (Resolve keeps only one marker per frame)
    `words` are the start times of every sung word, including ones dropped as too close to
    become notes: a word never runs past the next one (else a hold spans the words after it)."""
    hop = 512
    rms = librosa.feature.rms(y=librosa.to_mono(vocals), hop_length=hop)[0]
    rms = np.convolve(rms, np.ones(5) / 5, mode="same")      # ~60 ms smoothing
    db = librosa.amplitude_to_db(rms, ref=np.max)
    rate = sr / hop
    frame = lambda t: min(len(db) - 1, max(0, int(t * rate)))
    attack = frame(0.25) or 1
    settle, sustain_from, sustain_to = int(0.3 * rate), int(0.15 * rate), int(0.6 * rate)

    bounds = np.sort(np.array([it["time"] for it in items] + list(words or []), dtype=float))
    for i, it in enumerate(items):
        start = frame(it["time"])
        later = bounds[bounds > it["time"] + 1e-6]
        nxt = float(later[0]) if len(later) else None
        stop = frame(nxt) if nxt is not None else len(db) - 1
        peak = db[start:start + attack].max(initial=-80.0)
        quiet = np.nonzero(db[start + 1:stop] < peak - drop_db)[0]
        end = start + 1 + quiet[0] if len(quiet) else stop
        if end - start > sustain_to:
            level = np.median(db[start + sustain_from:start + sustain_to])
            faded = np.nonzero(db[start + settle:end] < level - sustain_drop_db)[0]
            if len(faded):
                end = start + settle + faded[0]
        seconds = (end - start) / rate
        if nxt is not None:
            seconds = min(seconds, nxt - it["time"] - gap_frames / FPS)
        it["duration"] = max(seconds, 0.0)
    return items


def syllables(word: str) -> int:
    """Syllables in a lyric word: pronunciation dictionary, then sung forms
    ("tryin" -> "trying"), then counting vowel groups for made-up words ("eenie")."""
    import pronouncing
    w = word.lower().strip("'")
    for candidate in (w, w + "g" if w.endswith("in") else None):
        if candidate:
            phones = pronouncing.phones_for_word(candidate)
            if phones:
                return pronouncing.syllable_count(phones[0])
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups) - (1 if w.endswith("e") and not w.endswith(("le", "ee", "ie")) and len(groups) > 1 else 0)
    return max(n, 1)


def split_multisyllable_holds(items: list[dict], vocals: np.ndarray, sr: int, hold_min: float,
                              min_gap: float) -> list[dict]:
    """A drawn-out word with 2+ syllables ("beautiful", "amazing") becomes taps instead of one
    long note, the way it's charted by hand: a tap on the word plus taps on its strongest vocal
    onsets, about one per two syllables (hand charts tap 2-3 syllable words twice).
    Words sung at least SPLIT_FRACTION of the hold length qualify."""
    hop = 256
    env = librosa.onset.onset_strength(y=librosa.to_mono(vocals), sr=sr, hop_length=hop)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=0.1, wait=5)
    peak_t = peaks * hop / sr
    out = []
    for i, it in enumerate(items):
        out.append(it)
        n = syllables(it["word"])
        if it["duration"] < hold_min * SPLIT_FRACTION or n < 2:
            continue
        extra = max(1, n // 2)
        end = it["time"] + it["duration"]
        inside = [(env[p], t) for p, t in zip(peaks, peak_t) if it["time"] + min_gap <= t <= end - 0.05]
        chosen = []
        for _, t in sorted(inside, reverse=True):
            if len(chosen) == extra:
                break
            if all(abs(t - c) >= min_gap for c in chosen):
                chosen.append(t)
        if not chosen:
            continue                      # no clear syllable onsets: keep it as a hold
        it["duration"] = 0.0              # now a tap, like each syllable after it
        for t in sorted(chosen):
            out.append({**it, "time": float(t), "aligned": float(t), "snapped": True, "duration": 0.0,
                        "syllable": True})
    return out


def drop_crowded(items: list[dict], min_gap: float) -> list[dict]:
    kept, last = [], -np.inf
    for item in items:
        if item["time"] - last >= min_gap:
            kept.append(item)
            last = item["time"]
    return kept


# ------------------------------------------------------------------ output

def seconds_to_tc(seconds: float) -> str:
    # Non-drop timecode counts whole frames at the rounded rate (30 for 29.97)
    frames = max(0, int(round(seconds * FPS)))
    secs, ff = divmod(frames, round(FPS))
    mins, ss = divmod(secs, 60)
    hh, mm = divmod(mins, 60)
    return f"{hh + TC_START_HOURS:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def build_markers(items: list[dict]) -> list[dict]:
    """One marker per tap; a start and an end marker per held note (the game
    reads held notes as consecutive start/end pairs)."""
    markers = []
    for it in items:
        if it["held"]:
            markers.append({**it, "edge": "start"})
            markers.append({**it, "edge": "end", "time": it["time"] + it["duration"]})
        else:
            markers.append({**it, "edge": ""})
    return markers


def write_edl(path: Path, markers: list[dict], title: str):
    """DaVinci Resolve: Timeline > Import > Timeline Markers from EDL.
    The timeline must start at 01:00:00:00 and run at FPS.
    Colors match the level files: blue = tap, purple = held start/end."""
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    for i, m in enumerate(sorted(markers, key=lambda m: m["time"]), 1):
        color = "ResolveColorPurple" if m["edge"] else "ResolveColorBlue"
        name = f"{m['word']} {m['edge']}".strip()
        tc_in = seconds_to_tc(m["time"])
        tc_out = seconds_to_tc(m["time"] + 1 / FPS)
        lines.append(f"{i:03d}  001      V     C        {tc_in} {tc_out} {tc_in} {tc_out}  ")
        lines.append(f" |C:{color} |M:{name} |D:1")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def line_spans(lines: list[dict], items: list[dict], lead: float = 0.1,
               linger: float = 2.0) -> list[dict]:
    """What's on screen for each lyric line. A line is a list of display units,
    one per word; hyphenated words are separate words ("eenie-meenie" -> "eenie",
    "meenie"). Ad-libs in (parentheses) are left out, and punctuation is stripped.
    A line shows just before its first unit and stays until the next line
    starts (or `linger` seconds after its last unit, or as long as that word is held:
    a held note never outlasts its word on screen)."""
    spans, i = [], 0
    for line in lines:
        n = len(line["words"])
        units = []
        for (word, _, adlib), it in zip(line["pieces"], items[i:i + n]):
            text = clean_word(word)
            if not adlib and text:
                units.append({"start": it["time"], "text": text, "sung": it.get("duration", 0.0)})
        i += n
        if units:
            spans.append({"units": units, "text": " ".join(u["text"] for u in units)})

    # A line appears `lead` before its first word, but never before the previous line's
    # last word has started (when one line runs straight into the next)
    for k, s in enumerate(spans):
        prev_last = spans[k - 1]["units"][-1]["start"] + 0.05 if k else 0.0
        s["start"] = max(0.0, s["units"][0]["start"] - lead, min(prev_last, s["units"][0]["start"]))
    for k, s in enumerate(spans):
        units = s["units"]
        nxt = spans[k + 1]["start"] if k + 1 < len(spans) else np.inf
        last = units[-1]["start"]
        s["end"] = max(min(last + linger, nxt), last + units[-1]["sung"], s["start"] + 0.3)
        for j, u in enumerate(units):
            end = units[j + 1]["start"] if j + 1 < len(units) else s["end"]
            u["end"] = max(end, u["start"] + 0.05)
    return spans


def write_srt(path: Path, spans: list[dict], words_per_cue: int):
    """Subtitles showing `words_per_cue` words at a time, never spanning two
    lyric lines. 0 = the whole line at once."""
    cues = []
    for s in spans:
        if words_per_cue <= 0:
            cues.append((s["start"], s["end"], s["text"]))
            continue
        units = s["units"]
        for k in range(0, len(units), words_per_cue):
            group = units[k:k + words_per_cue]
            cues.append((group[0]["start"], group[-1]["end"],
                         " ".join(u["text"] for u in group)))
    out = [f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n"
           for n, (a, b, text) in enumerate(cues, 1)]
    path.write_text("\n".join(out), encoding="utf-8")


# --------------------------------------------------------------- lyric video

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Fill,Arial,84,&H0000D7FF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,2,80,80,110,1
Style: Box,Arial,84,&H00FFFFFF,&H00FFFFFF,&H00E06A3A,&H00000000,-1,0,0,0,100,100,0,0,3,10,0,2,80,80,110,1
Style: Plain,Arial,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,2,80,80,110,1
Style: Title,Arial,64,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,2,0,1,3,0,8,80,80,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    s, cs = divmod(cs, 100)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def chunk_spans(spans: list[dict], n: int) -> list[dict]:
    """Splits each lyric line into on-screen pieces of `n` words (0 = whole line),
    the same way the subtitles are split. Each piece shows from its first word
    until the next piece starts (the first keeps the line's lead-in)."""
    if n <= 0:
        return spans
    out = []
    for s in spans:
        units = s["units"]
        groups = [units[k:k + n] for k in range(0, len(units), n)]
        for g, group in enumerate(groups):
            start = s["start"] if g == 0 else group[0]["start"]
            end = groups[g + 1][0]["start"] if g + 1 < len(groups) else s["end"]
            out.append({"units": group, "text": " ".join(u["text"] for u in group),
                        "start": start, "end": max(end, start + 0.05)})
    return out


def write_ass(path: Path, spans: list[dict], style: str, duration: float,
              title: str = "", artist: str = "", words_per_line: int = 0, last_word: float = 0.6):
    """fill:  each word fills with colour as it is sung (ASS \\kf karaoke).
    box:   a box highlights the word currently being sung.
    plain: just the lyric lines, no highlight.
    The last word of a line is highlighted for at most `last_word` seconds.
    title / artist, if given, show at the top for the whole song."""
    events = []
    end = ass_time(duration)
    # Name and artist are one event on two lines, so a name long enough to wrap
    # pushes the artist down instead of running into it
    header = []
    if title:
        header.append(ass_escape(title))
    if artist:
        header.append(f"{{\\fs40\\b0\\bord2\\c&HB4B4B4&}}{ass_escape(artist)}")
    if header:
        events.append(f"Dialogue: 1,0:00:00.00,{end},Title,,0,0,0,,{{\\fad(800,800)}}" + "\\N".join(header))
    for s in chunk_spans(spans, words_per_line):
        units = s["units"]
        starts = [u["start"] for u in units]
        last_end = min(starts[-1] + last_word, s["end"])
        if style == "plain":
            events.append(f"Dialogue: 0,{ass_time(s['start'])},{ass_time(s['end'])},Plain,,0,0,0,,"
                          f"{ass_escape(s['text'])}")
        elif style == "fill":
            # \kf durations are relative to the event start, in centiseconds.
            # Work from absolute boundaries so rounding doesn't drift.
            cs = [int(round(b * 100)) for b in [s["start"]] + starts + [last_end]]
            text = f"{{\\k{cs[1] - cs[0]}}}"
            for j, u in enumerate(units):
                space = " " if j + 1 < len(units) else ""
                text += f"{{\\kf{cs[j + 2] - cs[j + 1]}}}{ass_escape(u['text'])}{space}"
            events.append(f"Dialogue: 0,{ass_time(s['start'])},{ass_time(s['end'])},Fill,,0,0,0,,{text}")
        else:
            # One event per word; only the current word's box is opaque.
            # Before the first word and after the last, no word is boxed.
            edges = [s["start"]] + starts + [last_end, s["end"]]
            for j in range(len(edges) - 1):
                if edges[j + 1] - edges[j] < 0.01:
                    continue
                current = j - 1                       # -1 = lead-in, len = tail
                text = ""
                for k, u in enumerate(units):
                    box = "&H00&" if k == current else "&HFF&"
                    space = " " if k + 1 < len(units) else ""
                    text += f"{{\\3a{box}}}{ass_escape(u['text'])}{{\\3a&HFF&}}{space}"
                events.append(f"Dialogue: 0,{ass_time(edges[j])},{ass_time(edges[j + 1])},"
                              f"Box,,0,0,0,,{text}")
    path.write_text(ASS_HEADER + "\n".join(events) + "\n", encoding="utf-8")


def write_lyrics_json(path: Path, spans: list[dict], style: str, words_per_line: int,
                      title: str = "", artist: str = ""):
    """Lyrics for the game to show itself (lyrics.json in a level folder, read by LyricsDisplay).
    Lines are split into words_per_line chunks the same way as the subtitles and lyric video."""
    r = lambda t: round(float(t), 3)
    lines = [{"start": r(s["start"]), "end": r(s["end"]),
              "words": [{"text": u["text"], "start": r(u["start"]), "end": r(u["end"])} for u in s["units"]]}
             for s in chunk_spans(spans, words_per_line)]
    data = {"title": title, "artist": artist, "style": style, "lines": lines}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def render_lyric_video(ass_path: Path, video_path: Path, duration: float, audio: Path | None = None):
    """White lyrics on black, 1920x1080 at FPS, with the song's audio if given.
    The filter graph can't take Windows paths or names with commas, so render
    from a plain-named copy. FFMPEG (set by the game) overrides ffmpeg on PATH."""
    import os
    import shutil
    import subprocess
    import tempfile
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("FFmpeg not found (needed to render the lyric video).")
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(ass_path, Path(tmp) / "lyrics.ass")
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:r={Fraction(FPS).limit_denominator(1001)}:d={duration:.3f}"]
        if audio:
            cmd += ["-i", str(Path(audio).resolve()), "-map", "0:v", "-map", "1:a",
                    "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-vf", "ass=lyrics.ass",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                str(video_path.resolve())]
        subprocess.run(cmd, cwd=tmp, check=True)


def write_csv(path: Path, markers: list[dict]):
    header = ["#", "Word", "Seconds", "Aligned", "Snapped", "Confidence",
              "Duration", "Edge", "", "", "Record In", "Pitch", "Height"]
    assert header[TIME_COLUMN] == "Record In"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, m in enumerate(markers, 1):
            w.writerow([i, m["word"], f"{m['time']:.3f}", f"{m['aligned']:.3f}",
                        "yes" if m["snapped"] else "no", f"{m['confidence']:.2f}",
                        f"{m['duration']:.2f}", m["edge"], "", "",
                        seconds_to_tc(m["time"]),
                        fmt(m.get("pitch", float("nan")), 2), fmt(m.get("height", float("nan")), 3)])


# -------------------------------------------------------------------- main

def standard_fps(fps: float) -> float:
    """Snaps a measured rate (29.97003) to the standard one it stands for (30000/1001)."""
    for r in STANDARD_FPS:
        if abs(fps - float(r)) < 0.02:
            return float(r)
    return fps if fps > 1 else 30.0


def video_fps(path: Path) -> float | None:
    """Frame rate of a video file (None for audio-only files or without ffprobe)."""
    import os
    import shutil
    import subprocess
    ffmpeg = os.environ.get("FFMPEG")
    probe = str(Path(ffmpeg).with_name("ffprobe.exe")) if ffmpeg else None
    if not probe or not Path(probe).exists():
        probe = shutil.which("ffprobe")
    if not probe:
        return None
    try:
        out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(Fraction(out)) if out else None
    except (OSError, ValueError, ZeroDivisionError, subprocess.SubprocessError):
        return None


def ask_words_per_subtitle(default: int = 1) -> int:
    if not sys.stdin.isatty():                       # e.g. launched by another program
        return default
    while True:
        ans = input(f"Words per subtitle? (1 = one word at a time, 0 = whole lyric line) [{default}]: ").strip()
        if not ans:
            return default
        if ans.isdigit():
            return int(ans)
        print("Please enter a whole number, like 1, 3 or 0.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Auto-chart a song from audio + lyrics.")
    p.add_argument("song", type=Path, help="audio file (mp3/wav/flac/ogg)")
    p.add_argument("lyrics", type=Path, help="plain-text lyrics file")
    p.add_argument("-o", "--out", type=Path, help="output CSV (default: <song>_notes.csv)")
    p.add_argument("--model", default="facebook/wav2vec2-large-960h-lv60-self",
                   help="HuggingFace wav2vec2 CTC model (English by default)")
    p.add_argument("--snap-window", type=float, default=0.15,
                   help="max seconds a word may move to reach an onset")
    p.add_argument("--min-gap", type=float, default=0.12,
                   help="minimum seconds between targets; closer ones are dropped")
    p.add_argument("--offset", type=float, default=0.0,
                   help="shift every target by this many seconds (calibration)")
    p.add_argument("--hold-min", type=float, default=1.0,
                   help="words sung at least this many seconds are flagged as possible held notes")
    p.add_argument("--words", type=int, metavar="N",
                   help="words per subtitle (0 = whole lyric line); asked when run if not given")
    p.add_argument("--karaoke", choices=["fill", "box", "plain"], default="fill",
                   help="lyric video style: words fill with colour, a box jumps word to word, "
                        "or plain lines with no highlight")
    p.add_argument("--no-video", action="store_true",
                   help="skip rendering the lyric video (still writes the .ass)")
    p.add_argument("--no-audio", action="store_true",
                   help="leave the song's audio out of the lyric video")
    p.add_argument("--title", default="", help="song name shown at the top of the lyric video")
    p.add_argument("--artist", default="", help="artist shown under the song name")
    p.add_argument("--fps", type=float,
                   help="frame rate of the video / Resolve timeline (e.g. 29.97); "
                        "read from the song file if it's a video, otherwise 30")
    p.add_argument("--reseparate", action="store_true",
                   help="ignore the cached vocal stem and run Demucs again")
    args = p.parse_args(argv)

    if not args.song.is_file():
        sys.exit(f"Song not found: {args.song}")
    if not args.lyrics.is_file():
        sys.exit(f"Lyrics not found: {args.lyrics}")

    lines = parse_lyrics(args.lyrics.read_text(encoding="utf-8"))
    words = [w for line in lines for w in line["words"]]
    if not words:
        sys.exit("No words found in the lyrics file.")
    if args.words is None:
        args.words = ask_words_per_subtitle()
    global FPS
    FPS = standard_fps(args.fps or video_fps(args.song) or 30.0)
    print(f"Timeline frame rate: {FPS:.3f} fps")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    stems, sr = load_or_separate(args.song, device, args.reseparate)
    vocals = stems["vocals"]
    aligned = align_words(vocals, sr, words, args.model, device)
    onsets = vocal_onsets(vocals, sr)

    # Ad-libs (words in parentheses) are aligned for timing but never become notes,
    # and don't get to claim vocal onsets the main words need.
    items = aligned
    adlibs = [adlib for line in lines for _, _, adlib in line["pieces"]]
    for it, adlib in zip(items, adlibs):
        it["adlib"] = adlib
        it["time"], it["snapped"] = it["aligned"], False
    main_words = [it for it in items if not it["adlib"]]
    snap_to_onsets(main_words, onsets, args.snap_window, args.min_gap)
    kept = drop_crowded(main_words, args.min_gap)
    estimate_durations(kept, vocals, sr, words=[it["time"] for it in main_words])
    crowded = len(main_words) - len(kept)
    kept = split_multisyllable_holds(kept, vocals, sr, args.hold_min, args.min_gap)
    split_taps = sum(1 for it in kept if it.get("syllable"))
    for it in items + [it for it in kept if it.get("syllable")]:
        it["time"] += args.offset
    for it in kept:
        it["held"] = it["duration"] >= args.hold_min

    # Sung pitch of each note -> its height on screen in the game (higher = further up)
    track = pitch_track(vocals, sr)
    for it in kept:
        it["pitch"] = note_pitch(track, it["time"] - args.offset, it["duration"])
    for it, h in zip(kept, heights([it["pitch"] for it in kept])):
        it["height"] = h

    markers = build_markers(kept)
    taps = [m for m in markers if not m["edge"]]
    held = [m for m in markers if m["edge"]]

    out = args.out or args.song.with_name(args.song.stem + "_notes.csv")
    held_out = out.with_name(out.stem.removesuffix("_notes") + "_held.csv")
    edl = out.with_suffix(".edl")
    out.parent.mkdir(parents=True, exist_ok=True)
    base = out.stem.removesuffix("_notes")
    srt = out.with_name(base + "_lyrics.srt")
    ass = out.with_name(base + "_lyrics.ass")
    lyrics_json = out.with_name(base + "_lyrics.json")
    video = out.with_name(base + "_lyrics.mp4")
    drums_out = out.with_name(base + "_drums.csv")
    instrumental = out.with_name(base + "_instrumental.wav")
    write_csv(out, taps)
    write_csv(held_out, held)
    write_edl(edl, markers, args.song.stem)
    hits = drum_hits(stems["drums"], sr)
    write_drums_csv(drums_out, hits)
    sf.write(str(instrumental), np.clip(stems["instrumental"], -1, 1).T, sr)
    spans = line_spans(lines, items)
    write_srt(srt, spans, args.words)
    duration = len(vocals[0]) / sr
    write_ass(ass, spans, args.karaoke, duration, args.title.strip(), args.artist.strip(), args.words)
    write_lyrics_json(lyrics_json, spans, args.karaoke, args.words, args.title.strip(), args.artist.strip())
    if not args.no_video:
        print("Rendering lyric video...")
        render_lyric_video(ass, video, duration, None if args.no_audio else args.song)

    snapped = sum(it["snapped"] for it in kept)
    low = sum(it["confidence"] < 0.2 for it in kept)
    print(f"\n{len(words)} words -> {len(taps)} taps + {len(held) // 2} held notes "
          f"({snapped} snapped to onsets, {sum(adlibs)} ad-libs skipped, "
          f"{crowded} dropped as too close, {split_taps} syllable taps added)")
    if low:
        print(f"{low} targets have low alignment confidence; check the Confidence column.")
    print(f"Wrote {out}  (taps)")
    print(f"Wrote {held_out}  (held start/end pairs, >= {args.hold_min}s)")
    print(f"Wrote {edl}  (Resolve markers: blue = tap, purple = held start/end)")
    kinds = {k: sum(1 for _, kind, _ in hits if kind == k) for k in ("kick", "snare", "hihat")}
    print(f"Wrote {drums_out}  ({len(hits)} drum hits: {kinds['kick']} kick, {kinds['snare']} snare, "
          f"{kinds['hihat']} hi-hat)")
    print(f"Wrote {instrumental}  (the song without vocals)")
    shown = "whole lyric lines" if args.words <= 0 else f"{args.words} word(s) at a time"
    print(f"Wrote {srt}  (lyrics subtitles, {shown})")
    print(f"Wrote {ass}  (karaoke lyrics, {args.karaoke} style)")
    print(f"Wrote {lyrics_json}  (lyrics for the game to show)")
    if not args.no_video:
        audio_note = "no audio" if args.no_audio else "with audio"
        print(f"Wrote {video}  (lyric video, {audio_note}; put it at the start of the timeline)")


if __name__ == "__main__":
    main()
