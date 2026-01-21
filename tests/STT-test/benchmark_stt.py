#!/usr/bin/env python3
"""Benchmark Whisper STT (og valgfritt møtereferat via Ollama) på en lokal lydfil.

Eksempler:
  python benchmark_stt.py --file "C:\\path\\to\\audio.mp3" --model turbo --language no --no-summarize
  python benchmark_stt.py --file ./audio.mp3 --summarize

Tips:
- Whisper trenger vanligvis ffmpeg installert for mp3/webm.
- Hvis du vil måle "cold start", kjør scriptet på nytt.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import whisper

# Valgfritt: møtereferat via Ollama
try:
    from app.pipeline.summarize import create_meeting_minutes
except Exception:  # pragma: no cover
    create_meeting_minutes = None


def format_transcription(text: str) -> str:
    text = text.replace(". ", ".\n")
    text = text.replace("? ", "?\n")
    text = text.replace("! ", "!\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path til lydfil (mp3/wav/m4a/webm/...) ")
    parser.add_argument("--model", default=os.getenv("WHISPER_MODEL", "turbo"), help="Whisper-modell (f.eks. tiny/base/small/medium/large/turbo)")
    parser.add_argument("--language", default="no", help="Språkkode (default: no)")
    parser.add_argument("--summarize", dest="summarize", action="store_true", help="Lag møtereferat via Ollama/Mistral")
    parser.add_argument("--no-summarize", dest="summarize", action="store_false", help="Ikke lag møtereferat")
    parser.set_defaults(summarize=False)
    parser.add_argument("--json", action="store_true", help="Skriv kun JSON til stdout")

    args = parser.parse_args()

    audio_path = Path(args.file).expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Fant ikke fil: {audio_path}")

    t0 = time.perf_counter()
    model = whisper.load_model(args.model)
    model_load_s = time.perf_counter() - t0

    # Varighet
    audio_duration_s = None
    try:
        audio = whisper.load_audio(str(audio_path))
        audio_duration_s = len(audio) / 16000.0
    except Exception:
        pass

    t1 = time.perf_counter()
    result = model.transcribe(str(audio_path), language=args.language)
    transcribe_s = time.perf_counter() - t1

    transcription = format_transcription(result.get("text", ""))

    meeting_minutes = ""
    summarize_s = 0.0
    if args.summarize:
        if create_meeting_minutes is None:
            meeting_minutes = "(Summarize er aktivert, men app.pipeline.summarize kunne ikke importeres.)"
        else:
            t2 = time.perf_counter()
            meeting_minutes = create_meeting_minutes(transcription)
            summarize_s = time.perf_counter() - t2

    timings = {
        "model_load_seconds": round(model_load_s, 4),
        "transcribe_seconds": round(transcribe_s, 4),
        "summarize_seconds": round(summarize_s, 4),
        "total_seconds": round(transcribe_s + summarize_s, 4),
    }
    if audio_duration_s is not None and audio_duration_s > 0:
        timings["audio_duration_seconds"] = round(audio_duration_s, 4)
        timings["real_time_factor"] = round(transcribe_s / audio_duration_s, 4)

    out = {
        "whisper_model": args.model,
        "file": str(audio_path),
        "language": args.language,
        "timings": timings,
        "transcription": transcription,
        "meeting_minutes": meeting_minutes,
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("=== TIMING ===")
        for k, v in timings.items():
            print(f"{k}: {v}")
        print("\n=== TRANSKRIPSJON ===\n")
        print(transcription)
        if args.summarize:
            print("\n=== MØTEREFERAT ===\n")
            print(meeting_minutes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
