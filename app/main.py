from __future__ import annotations

import os
import pathlib
import torch
import whisper
import soundfile as sf

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pyannote.audio import Pipeline

from app.pipeline.summarize import create_meeting_minutes
from app.utils.meeting import create_meeting


# =============================
# Konfig (fast, ingen valg)
# =============================
WHISPER_MODEL = "turbo"
WHISPER_LANGUAGE = "no"
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"

CHUNK_SECONDS = 600.0      # 10 minutter
OVERLAP_SECONDS = 30.0     # 30 sek overlapp


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def format_text(text: str) -> str:
    text = text.replace(". ", ".\n").replace("? ", "?\n").replace("! ", "!\n")
    return "\n".join(ln.strip() for ln in text.split("\n") if ln.strip())


# =============================
# Audio helpers
# =============================
def load_waveform(audio_path: pathlib.Path):
    wav, sr = sf.read(str(audio_path))
    if wav.ndim == 1:
        wav = wav[None, :]
    else:
        wav = wav.T
    return torch.from_numpy(wav).float(), int(sr)


def slice_waveform(waveform, sr, start_s, end_s):
    start = int(start_s * sr)
    end = int(end_s * sr)
    return waveform[:, start:end]


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


# =============================
# Long diarization (chunk + stitch)
# =============================
def diarize_long(pipeline: Pipeline, audio_path: pathlib.Path):
    waveform, sr = load_waveform(audio_path)
    total_s = waveform.shape[1] / sr

    global_counter = 1
    global_segments = []
    recent_segments = []

    t = 0.0
    while t < total_s:
        start_s = max(0.0, t - OVERLAP_SECONDS)
        end_s = min(total_s, t + CHUNK_SECONDS)

        chunk_wav = slice_waveform(waveform, sr, start_s, end_s)
        diar = pipeline({"waveform": chunk_wav, "sample_rate": sr})

        chunk_segments = []
        for seg, _, spk in diar.itertracks(yield_label=True):
            gs = float(seg.start) + start_s
            ge = float(seg.end) + start_s
            chunk_segments.append({"start": gs, "end": ge, "speaker": str(spk)})

        for seg in chunk_segments:
            best_match = None
            best_ov = 0.0

            for r in recent_segments:
                ov = overlap(seg["start"], seg["end"], r["start"], r["end"])
                if ov > best_ov:
                    best_ov = ov
                    best_match = r["speaker"]

            if best_match and best_ov >= 1.0:
                seg["speaker"] = best_match
            else:
                seg["speaker"] = f"Person {global_counter}"
                global_counter += 1

        global_segments.extend(chunk_segments)

        recent_segments = [
            s for s in global_segments
            if s["end"] >= (end_s - OVERLAP_SECONDS * 2)
        ]

        t += CHUNK_SECONDS

    return global_segments


# =============================
# Align Whisper → Speaker
# =============================
def label_whisper_segments(whisper_segments, diar_segments):
    labeled = []

    for ws in whisper_segments:
        best_spk = "Ukjent"
        best_ov = 0.0

        for ds in diar_segments:
            ov = overlap(ws["start"], ws["end"], ds["start"], ds["end"])
            if ov > best_ov:
                best_ov = ov
                best_spk = ds["speaker"]

        labeled.append(
            {
                "speaker": best_spk,
                "text": ws["text"].strip(),
            }
        )

    return labeled


def render_transcription(labeled):
    return format_text(
        "\n".join(
            f"{s['speaker']}: {s['text']}"
            for s in labeled
            if s["text"]
        )
    )


# =============================
# App
# =============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = pick_device()
whisper_model = whisper.load_model(WHISPER_MODEL, device=device)
diar_pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    meeting_id, meeting_dir = create_meeting()

    audio_path = meeting_dir / "audio.wav"
    audio_path.write_bytes(await file.read())

    # Whisper
    result = await run_in_threadpool(
        whisper_model.transcribe,
        str(audio_path),
        language=WHISPER_LANGUAGE,
        verbose=False,
    )

    whisper_segments = result["segments"]

    # Diarization (long-form safe)
    diar_segments = await run_in_threadpool(
        diarize_long, diar_pipeline, audio_path
    )

    labeled = label_whisper_segments(whisper_segments, diar_segments)
    transcription = render_transcription(labeled)

    minutes = await run_in_threadpool(
        create_meeting_minutes, transcription
    )

    (meeting_dir / "transcription.txt").write_text(transcription, encoding="utf-8")
    (meeting_dir / "summary.txt").write_text(minutes, encoding="utf-8")

    return {
        "meeting_id": meeting_id,
        "transcription": transcription,
        "meeting_minutes": minutes,
    }
