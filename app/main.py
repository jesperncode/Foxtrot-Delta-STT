from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import pathlib

from app.pipeline.summarize import create_meeting_minutes
from app.utils.meeting import create_meeting


def format_transcription(text: str) -> str:
    text = text.replace(". ", ".\n")
    text = text.replace("? ", "?\n")
    text = text.replace("! ", "!\n")

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|\[::1\]):\d+$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("turbo")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    content = await file.read()

    meeting_id, meeting_dir = create_meeting()

    suffix = pathlib.Path(file.filename).suffix or ".wav"
    audio_path = meeting_dir / f"audio{suffix}"

    with open(audio_path, "wb") as f:
        f.write(content)

    # Transkribere
    result = model.transcribe(str(audio_path), language="no")
    transcription = format_transcription(result["text"])

    # Lag referat
    meeting_minutes = create_meeting_minutes(transcription)

    # Lagre output
    (meeting_dir / "transcription.txt").write_text(
        transcription, encoding="utf-8"
    )
    (meeting_dir / "summary.txt").write_text(
        meeting_minutes, encoding="utf-8"
    )

    return {
        "transcription": transcription,
        "meeting_minutes": meeting_minutes
    }
