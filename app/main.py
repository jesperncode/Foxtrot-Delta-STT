# app/main.py
from __future__ import annotations

import os
import pathlib
from collections import defaultdict

import torch
import whisper
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from pyannote.audio import Pipeline

from app.pipeline.summarize import create_meeting_minutes
from app.utils.meeting import create_meeting


# ==========================================================
# KONFIG
# ==========================================================
WHISPER_MODEL = "turbo"
WHISPER_LANGUAGE = "no"

PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
PYANNOTE_SNAPSHOT_DIR = os.getenv("PYANNOTE_SNAPSHOT_DIR")

# Guardrail: stopper eksplosjon i antall speakers
PYANNOTE_MAX_SPEAKERS = int(os.getenv("PYANNOTE_MAX_SPEAKERS", "6"))

# Default cleanup (møter)
MEETING_MIN_TOTAL_SPEECH_S = float(os.getenv("MEETING_MIN_TOTAL_SPEECH_S", "12"))
MEETING_KEEP_TOP_SPEAKERS = int(os.getenv("MEETING_KEEP_TOP_SPEAKERS", "8"))

# Hard cleanup (podcast/intervju)
PODCAST_MIN_TOTAL_SPEECH_S = float(os.getenv("PODCAST_MIN_TOTAL_SPEECH_S", "25"))
PODCAST_KEEP_TOP_SPEAKERS = int(os.getenv("PODCAST_KEEP_TOP_SPEAKERS", "2"))

MERGE_GAP_S = float(os.getenv("PYANNOTE_MERGE_GAP_S", "0.4"))


# ==========================================================
# DEVICE
# ==========================================================
def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = pick_device()
print(f"[INFO] Torch device: {DEVICE}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")


# ==========================================================
# MODELLER
# ==========================================================
whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)

if PYANNOTE_SNAPSHOT_DIR:
    diar_pipeline = Pipeline.from_pretrained(PYANNOTE_SNAPSHOT_DIR)
else:
    diar_pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL)


# ==========================================================
# HJELPERE
# ==========================================================
def format_text(text: str) -> str:
    text = text.replace(". ", ".\n").replace("? ", "?\n").replace("! ", "!\n")
    return "\n".join(l.strip() for l in text.splitlines() if l.strip())


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def is_podcast_like(text: str) -> bool:
    t = (text or "").lower()
    keys = ["podcast", "episode", "i studio", "velkommen til", "du lytter"]
    hits = sum(1 for k in keys if k in t)
    return hits >= 2


def load_waveform(audio_path: pathlib.Path) -> tuple[torch.Tensor, int]:
    # soundfile -> (samples,) eller (samples, channels)
    wav, sr = sf.read(str(audio_path), always_2d=False)
    if wav.ndim == 1:
        wav = wav[None, :]            # (1, T)
    else:
        wav = wav.T                   # (C, T)
    return torch.from_numpy(wav).float(), int(sr)


def ensure_annotation(diar):
    # 1) direkte Annotation
    if hasattr(diar, "itertracks"):
        return diar

    # 2) mapping-style (DiarizeOutput kan oppføre seg litt ulikt mellom versjoner)
    try:
        ann = diar["annotation"]
        if hasattr(ann, "itertracks"):
            return ann
    except Exception:
        pass

    try:
        ann = diar.get("annotation")
        if hasattr(ann, "itertracks"):
            return ann
    except Exception:
        pass

    # 3) attributter
    for attr in ("annotation", "annotations"):
        if hasattr(diar, attr):
            ann = getattr(diar, attr)
            if hasattr(ann, "itertracks"):
                return ann

    # 4) scan vars()
    try:
        for v in vars(diar).values():
            if hasattr(v, "itertracks"):
                return v
    except Exception:
        pass

    raise TypeError(f"Kunne ikke hente Annotation fra diarization-output: {type(diar)}")


def merge_adjacent(segments: list[dict], gap_s: float) -> list[dict]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: (x["start"], x["end"]))
    out = [segments[0].copy()]
    for s in segments[1:]:
        prev = out[-1]
        if s["speaker"] == prev["speaker"] and s["start"] - prev["end"] <= gap_s:
            prev["end"] = max(prev["end"], s["end"])
        else:
            out.append(s.copy())
    return out


def compute_total_speech(segments: list[dict]) -> dict[str, float]:
    tot = defaultdict(float)
    for s in segments:
        tot[s["speaker"]] += max(0.0, float(s["end"]) - float(s["start"]))
    return dict(tot)


def nearest_big_speaker(seg: dict, big_segments: list[dict]) -> str | None:
    s0, s1 = float(seg["start"]), float(seg["end"])
    best = None
    best_dist = 1e18
    for b in big_segments:
        b0, b1 = float(b["start"]), float(b["end"])
        if s1 < b0:
            dist = b0 - s1
        elif b1 < s0:
            dist = s0 - b1
        else:
            dist = 0.0
        if dist < best_dist:
            best_dist = dist
            best = b["speaker"]
    return best


def cleanup_speakers(
    segments: list[dict],
    min_total_speech_s: float,
    keep_top_speakers: int,
    merge_gap_s: float,
) -> list[dict]:
    if not segments:
        return []

    segments = merge_adjacent(segments, gap_s=min(merge_gap_s, 0.3))

    totals = compute_total_speech(segments)
    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)

    keep = set([spk for spk, _ in ranked[:keep_top_speakers]])
    keep |= set([spk for spk, t in ranked if t >= min_total_speech_s])

    if not keep and ranked:
        keep = set([spk for spk, _ in ranked[:2]])

    big_segments = [s for s in segments if s["speaker"] in keep]
    if not big_segments:
        return merge_adjacent(segments, gap_s=merge_gap_s)

    cleaned = []
    for s in segments:
        if s["speaker"] in keep:
            cleaned.append(s)
        else:
            repl = nearest_big_speaker(s, big_segments)
            ns = s.copy()
            if repl is not None:
                ns["speaker"] = repl
            cleaned.append(ns)

    return merge_adjacent(cleaned, gap_s=merge_gap_s)


# ==========================================================
# DIARIZATION (ingen fil-IO i pyannote)
# ==========================================================
def diarize_whole(
    pipeline: Pipeline,
    audio_path: pathlib.Path,
    max_speakers: int,
    podcast_mode: bool,
):
    waveform, sr = load_waveform(audio_path)

    kwargs = {"min_speakers": 1, "max_speakers": max_speakers}

    # ✅ viktig: send waveform+sr -> slipper AudioDecoder
    diar_out = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
    diar = ensure_annotation(diar_out)

    raw_segments = []
    for seg, _, lbl in diar.itertracks(yield_label=True):
        raw_segments.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(lbl)})

    if podcast_mode:
        min_total = PODCAST_MIN_TOTAL_SPEECH_S
        keep_top = PODCAST_KEEP_TOP_SPEAKERS
    else:
        min_total = MEETING_MIN_TOTAL_SPEECH_S
        keep_top = MEETING_KEEP_TOP_SPEAKERS

    cleaned = cleanup_speakers(
        raw_segments,
        min_total_speech_s=min_total,
        keep_top_speakers=keep_top,
        merge_gap_s=MERGE_GAP_S,
    )

    # map til Person 1..N etter cleanup
    speaker_map: dict[str, str] = {}
    counter = 1
    for s in cleaned:
        raw = s["speaker"]
        if raw not in speaker_map:
            speaker_map[raw] = f"Person {counter}"
            counter += 1
        s["speaker"] = speaker_map[raw]

    return cleaned


# ==========================================================
# ALIGN Whisper → Speaker
# ==========================================================
def label_whisper_segments(whisper_segments, diar_segments):
    labeled = []
    for ws in whisper_segments:
        ws_start = float(ws.get("start", 0.0))
        ws_end = float(ws.get("end", 0.0))

        best_spk = "Ukjent"
        best_ov = 0.0
        for ds in diar_segments:
            ov = overlap(ws_start, ws_end, ds["start"], ds["end"])
            if ov > best_ov:
                best_ov = ov
                best_spk = ds["speaker"]

        labeled.append({"speaker": best_spk, "text": (ws.get("text") or "").strip()})
    return labeled


def render_transcription(labeled):
    return format_text("\n".join(f"{s['speaker']}: {s['text']}" for s in labeled if s["text"]))


# ==========================================================
# FASTAPI
# ==========================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        meeting_id, meeting_dir = create_meeting()
        audio_path = meeting_dir / "audio.wav"
        audio_path.write_bytes(await file.read())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Kunne ikke lagre fil: {e}")

    # Whisper
    result = await run_in_threadpool(
        whisper_model.transcribe,
        str(audio_path),
        language=WHISPER_LANGUAGE,
        verbose=False,
    )
    whisper_segments = result.get("segments", [])

    first_text = " ".join((s.get("text") or "") for s in whisper_segments[:50])
    podcast_mode = is_podcast_like(first_text)

    # Diarization
    diar_segments = await run_in_threadpool(
        diarize_whole,
        diar_pipeline,
        audio_path,
        PYANNOTE_MAX_SPEAKERS,
        podcast_mode,
    )

    labeled = label_whisper_segments(whisper_segments, diar_segments)
    transcription = render_transcription(labeled)

    minutes = await run_in_threadpool(create_meeting_minutes, transcription)

    (meeting_dir / "transcription.txt").write_text(transcription, encoding="utf-8")
    (meeting_dir / "summary.txt").write_text(minutes, encoding="utf-8")

    return {
        "meeting_id": meeting_id,
        "podcast_mode": podcast_mode,
        "transcription": transcription,
        "meeting_minutes": minutes,
    }
