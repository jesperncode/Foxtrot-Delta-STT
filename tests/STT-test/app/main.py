from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os, tempfile, pathlib
import time
from app.pipeline.summarize import create_meeting_minutes

def format_transcription(text: str) -> str:
    #  Sett inn linjeskift etter punktum
    text = text.replace(". ", ".\n")

    #  Sett inn linjeskift etter spørsmål
    text = text.replace("? ", "?\n")

    #  Sett inn linjeskift etter utrop
    text = text.replace("! ", "!\n")

    #  Trim ekstra whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    return "\n".join(lines)


app = FastAPI()

# CORS-oppsett
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|\[::1\]):\d+$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


_MODEL_NAME = os.getenv("WHISPER_MODEL", "turbo")

# Last inn modellen én gang ved oppstart (varm start). Vi måler også hvor lang tid det tar,
# siden du vil sammenligne hastighet på ulike maskiner.
_t0 = time.perf_counter()
model = whisper.load_model(_MODEL_NAME)
MODEL_LOAD_SECONDS = time.perf_counter() - _t0

@app.get("/")
def root():
    return {
        "status": "ok",
        "whisper_model": _MODEL_NAME,
        "model_load_seconds": round(MODEL_LOAD_SECONDS, 4),
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    summarize: bool = Query(True, description="Om true: lag møtereferat via Ollama/Mistral"),
    language: str = Query("no", description="Språkkode for Whisper, f.eks. 'no'"),
):
    content = await file.read()

    suffix = pathlib.Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        temp_path = tmp.name

    try:
        timings = {
            "model_load_seconds": round(MODEL_LOAD_SECONDS, 4),
        }

        # (Valgfritt) beregn varighet på lydfilen. Dette er nyttig for å se "real-time factor".
        # whisper.load_audio bruker ffmpeg under panseret for mp3/webm/etc.
        audio_duration_s = None
        try:
            audio = whisper.load_audio(temp_path)
            audio_duration_s = len(audio) / 16000.0
        except Exception:
            # Ikke kritisk – hvis ffmpeg mangler eller filen er rar.
            pass

        #  Transkribér med Whisper
        t_stt0 = time.perf_counter()
        result = model.transcribe(temp_path, language=language)
        timings["transcribe_seconds"] = round(time.perf_counter() - t_stt0, 4)

        transcription = format_transcription(result.get("text", ""))

        meeting_minutes = ""
        if summarize:
            t_sum0 = time.perf_counter()
            meeting_minutes = create_meeting_minutes(transcription)
            timings["summarize_seconds"] = round(time.perf_counter() - t_sum0, 4)
        else:
            timings["summarize_seconds"] = 0.0

        timings["total_seconds"] = round(timings["transcribe_seconds"] + timings["summarize_seconds"], 4)

        if audio_duration_s is not None and audio_duration_s > 0:
            timings["audio_duration_seconds"] = round(audio_duration_s, 4)
            timings["real_time_factor"] = round(timings["transcribe_seconds"] / audio_duration_s, 4)

        return {
            "transcription": transcription,
            "meeting_minutes": meeting_minutes,
            "timings": timings,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

