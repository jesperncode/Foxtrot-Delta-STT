from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os, tempfile, pathlib
from app.pipeline.summarize import create_meeting_minutes


app = FastAPI()

# CORS-oppsett
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
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

    suffix = pathlib.Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        temp_path = tmp.name

    try:
        #  Transkribér med Whisper 
        result = model.transcribe(temp_path)
        transcription = result["text"]

        #  Lag møtereferat med Mistral
        meeting_minutes = create_meeting_minutes(transcription)

        # Returner begge
        return {
            "transcription": transcription,
            "meeting_minutes": meeting_minutes
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

