from __future__ import annotations

import time
import pathlib

import whisper
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.pipeline.summarize import create_meeting_minutes
from app.utils.meeting import create_meeting


def format_transcription(text: str) -> str:
    text = text.replace(". ", ".\n")
    text = text.replace("? ", "?\n")
    text = text.replace("! ", "!\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = pick_device()
model = whisper.load_model("turbo", device=device)
print("Whisper requested device:", device)
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Model param device:", next(model.parameters()).device)



@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    t0_total = time.perf_counter()

    content = await file.read()

    try:
        meeting_id, meeting_dir = create_meeting()
    except RuntimeError as e:
    # USB mangler / policy matcher ikke
        raise HTTPException(status_code=503, detail=str(e))

    suffix = pathlib.Path(file.filename).suffix or ".wav"
    audio_path = meeting_dir / f"audio{suffix}"

    with open(audio_path, "wb") as f:
        f.write(content)

    # Transkribering
    t0_asr = time.perf_counter()
    try:
        result = await run_in_threadpool(
            model.transcribe,
            str(audio_path),
            language="no",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper transcribe feilet: {e}")

    t_asr = time.perf_counter() - t0_asr
    transcription = format_transcription(result.get("text", ""))

    # Oppsummering
    t0_sum = time.perf_counter()
    try:
        meeting_minutes = await run_in_threadpool(create_meeting_minutes, transcription)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama/oppsummering feilet: {e}")

    t_sum = time.perf_counter() - t0_sum
    t_total = time.perf_counter() - t0_total

    (meeting_dir / "transcription.txt").write_text(transcription, encoding="utf-8")
    (meeting_dir / "summary.txt").write_text(meeting_minutes, encoding="utf-8")

    return {
        "meeting_id": meeting_id,
        "transcription": transcription,
        "meeting_minutes": meeting_minutes,
        "timings_seconds": {
            "transcription": round(t_asr, 3),
            "summarization": round(t_sum, 3),
            "total": round(t_total, 3),
        },
    }
