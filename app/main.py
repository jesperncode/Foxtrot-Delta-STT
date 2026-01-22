# app/main.py
from __future__ import annotations

import os
import pathlib
import re
import time
from collections import defaultdict
from typing import Any

import torch
import whisper
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

try:
    from pyannote.audio import Pipeline  # type: ignore
except Exception:
    Pipeline = None

from app.pipeline.summarize import create_meeting_minutes
from app.utils.meeting import create_meeting


# ==========================================================
# KONFIG
# ==========================================================
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "turbo")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "no")

DIARIZATION_ENABLED = os.getenv("DIARIZATION_ENABLED", "1") == "1"
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1")
PYANNOTE_SNAPSHOT_DIR = os.getenv("PYANNOTE_SNAPSHOT_DIR")  # lokal snapshot-map

PYANNOTE_MAX_SPEAKERS = int(os.getenv("PYANNOTE_MAX_SPEAKERS", "6"))

MEETING_MIN_TOTAL_SPEECH_S = float(os.getenv("MEETING_MIN_TOTAL_SPEECH_S", "12"))
MEETING_KEEP_TOP_SPEAKERS = int(os.getenv("MEETING_KEEP_TOP_SPEAKERS", "8"))

PODCAST_MIN_TOTAL_SPEECH_S = float(os.getenv("PODCAST_MIN_TOTAL_SPEECH_S", "25"))
PODCAST_KEEP_TOP_SPEAKERS = int(os.getenv("PODCAST_KEEP_TOP_SPEAKERS", "2"))

MERGE_GAP_S = float(os.getenv("PYANNOTE_MERGE_GAP_S", "0.4"))


# ==========================================================
# DEVICE + MODELLER (last én gang)
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Torch device: {DEVICE}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)

diar_pipeline = None
if DIARIZATION_ENABLED and Pipeline is not None:
    try:
        if PYANNOTE_SNAPSHOT_DIR:
            diar_pipeline = Pipeline.from_pretrained(PYANNOTE_SNAPSHOT_DIR)
            print(f"[INFO] Loaded pyannote pipeline from snapshot dir: {PYANNOTE_SNAPSHOT_DIR}")
        else:
            diar_pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL)
            print(f"[INFO] Loaded pyannote pipeline from hub: {PYANNOTE_MODEL}")
    except Exception as e:
        diar_pipeline = None
        print(f"[WARN] Diarization disabled (pipeline load failed): {e}")


# ==========================================================
# MEETING-ID -> DIR cache (fikser 404)
# ==========================================================
MEETING_DIRS: dict[str, pathlib.Path] = {}  # fylles i /transcribe


def get_meeting_dir(meeting_id: str) -> pathlib.Path:
    d = MEETING_DIRS.get(meeting_id)
    if d and d.exists():
        return d
    # fallback: hvis server er restartet og cache er tom:
    # prøv å finne ved å søke i noen vanlige steder (uten å gjette for mye)
    raise HTTPException(
        status_code=404,
        detail="Ukjent meeting_id (server restartet eller møtemappe ikke funnet). Kjør /transcribe på nytt.",
    )


# ==========================================================
# HJELPERE (tekst)
# ==========================================================
def format_text(text: str) -> str:
    text = (text or "").replace(". ", ".\n").replace("? ", "?\n").replace("! ", "!\n")
    return "\n".join(l.strip() for l in text.splitlines() if l.strip())


def is_podcast_like(text: str) -> bool:
    t = (text or "").lower()
    keys = ["podcast", "episode", "i studio", "velkommen til", "du lytter"]
    hits = sum(1 for k in keys if k in t)
    return hits >= 2


# ==========================================================
# HJELPERE (audio/diarization)
# ==========================================================
def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def load_waveform(audio_path: pathlib.Path) -> tuple[torch.Tensor, int]:
    wav, sr = sf.read(str(audio_path), always_2d=False)
    if wav.ndim == 1:
        wav = wav[None, :]
    else:
        wav = wav.T
    return torch.from_numpy(wav).float(), int(sr)


def ensure_annotation(diar: Any):
    if hasattr(diar, "itertracks"):
        return diar
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
    for attr in ("annotation", "annotations"):
        if hasattr(diar, attr):
            ann = getattr(diar, attr)
            if hasattr(ann, "itertracks"):
                return ann
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
    tot: dict[str, float] = defaultdict(float)
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

    cleaned: list[dict] = []
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


def diarize_whole(audio_path: pathlib.Path, max_speakers: int, podcast_mode: bool) -> list[dict]:
    if diar_pipeline is None:
        return []

    waveform, sr = load_waveform(audio_path)
    diar_out = diar_pipeline(
        {"waveform": waveform, "sample_rate": sr},
        min_speakers=1,
        max_speakers=max_speakers,
    )
    diar = ensure_annotation(diar_out)

    raw_segments: list[dict] = []
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

    speaker_map: dict[str, str] = {}
    counter = 1
    for s in cleaned:
        raw = s["speaker"]
        if raw not in speaker_map:
            speaker_map[raw] = f"Person {counter}"
            counter += 1
        s["speaker"] = speaker_map[raw]

    return cleaned


def label_whisper_segments(whisper_segments: list[dict], diar_segments: list[dict]) -> list[dict]:
    if not diar_segments:
        return [{"speaker": "Person 1", "text": (ws.get("text") or "").strip()} for ws in whisper_segments]

    labeled: list[dict] = []
    for ws in whisper_segments:
        ws_start = float(ws.get("start", 0.0))
        ws_end = float(ws.get("end", 0.0))

        best_spk = "Ukjent"
        best_ov = 0.0
        for ds in diar_segments:
            ov = overlap(ws_start, ws_end, float(ds["start"]), float(ds["end"]))
            if ov > best_ov:
                best_ov = ov
                best_spk = str(ds["speaker"])

        labeled.append({"speaker": best_spk, "text": (ws.get("text") or "").strip()})
    return labeled


def render_transcription(labeled: list[dict]) -> str:
    return format_text("\n".join(f"{s['speaker']}: {s['text']}" for s in labeled if s.get("text")))


def detect_speakers_in_text(transcription: str) -> list[str]:
    found = set(re.findall(r"\b(Person\s+\d+):", transcription))

    def key(x: str) -> int:
        m = re.search(r"(\d+)$", x)
        return int(m.group(1)) if m else 10**9

    return sorted(found, key=key)


def replace_speakers(text: str, mapping: dict[str, str]) -> str:
    out = text
    for speaker_id, name in mapping.items():
        name = (name or "").strip()
        if not name:
            continue
        out = out.replace(f"{speaker_id}:", f"{name}:")
    return out


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


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    t0_total = time.perf_counter()
    content = await file.read()

    # create_meeting() kan kaste hvis USB-policy ikke er oppfylt
    try:
        meeting_id, meeting_dir = create_meeting()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # cache meeting dir for senere PUT-kall
    MEETING_DIRS[meeting_id] = pathlib.Path(meeting_dir)

    suffix = pathlib.Path(file.filename).suffix or ".wav"
    audio_path = meeting_dir / f"audio{suffix}"
    audio_path.write_bytes(content)

    # Whisper
    t0_asr = time.perf_counter()
    result = await run_in_threadpool(
        whisper_model.transcribe,
        str(audio_path),
        language=WHISPER_LANGUAGE,
        verbose=False,
    )
    t1_asr = time.perf_counter()

    whisper_segments = result.get("segments", [])
    first_text = " ".join((s.get("text") or "") for s in whisper_segments[:50])
    podcast_mode = is_podcast_like(first_text)

    # Diarization (valgfri)
    t0_diar = time.perf_counter()
    diar_segments = await run_in_threadpool(
        diarize_whole,
        audio_path,
        PYANNOTE_MAX_SPEAKERS,
        podcast_mode,
    )
    t1_diar = time.perf_counter()

    labeled = label_whisper_segments(whisper_segments, diar_segments)
    transcription = render_transcription(labeled)

    # Oppsummering
    t0_sum = time.perf_counter()
    minutes = await run_in_threadpool(create_meeting_minutes, transcription)
    t1_sum = time.perf_counter()

    (meeting_dir / "transcription.txt").write_text(transcription, encoding="utf-8")
    (meeting_dir / "summary.txt").write_text(minutes, encoding="utf-8")

    t1_total = time.perf_counter()

    speakers = detect_speakers_in_text(transcription)
    speaker_objs = [{"speaker_id": s, "label": s, "name": ""} for s in speakers]

    return {
        "meeting_id": meeting_id,
        "podcast_mode": podcast_mode,
        "diarization_enabled": diar_pipeline is not None,
        "transcription": transcription,
        "meeting_minutes": minutes,
        "timings_seconds": {
            "transcription": round(t1_asr - t0_asr, 3),
            "diarization": round(t1_diar - t0_diar, 3),
            "summarization": round(t1_sum - t0_sum, 3),
            "total": round(t1_total - t0_total, 3),
        },
        "speakers": speaker_objs,
    }


@app.put("/meetings/{meeting_id}/speakers")
async def update_speakers(meeting_id: str, mapping: dict[str, str]):
    meeting_dir = get_meeting_dir(meeting_id)

    t_path = meeting_dir / "transcription.txt"
    s_path = meeting_dir / "summary.txt"

    if not t_path.exists() or not s_path.exists():
        raise HTTPException(status_code=404, detail="Mangler transkripsjon eller summary for dette møtet")

    transcription = t_path.read_text(encoding="utf-8")
    minutes = s_path.read_text(encoding="utf-8")

    transcription2 = replace_speakers(transcription, mapping)
    minutes2 = replace_speakers(minutes, mapping)

    t_path.write_text(transcription2, encoding="utf-8")
    s_path.write_text(minutes2, encoding="utf-8")

    speakers = [{"speaker_id": k, "label": k, "name": (v or "")} for k, v in mapping.items()]

    return {
        "transcription": transcription2,
        "meeting_minutes": minutes2,
        "speakers": speakers,
    }


@app.put("/meetings/{meeting_id}/texts")
async def update_texts(meeting_id: str, payload: dict):
    meeting_dir = get_meeting_dir(meeting_id)

    transcription = payload.get("transcription", "")
    minutes = payload.get("meeting_minutes", "")

    (meeting_dir / "transcription.txt").write_text(transcription, encoding="utf-8")
    (meeting_dir / "summary.txt").write_text(minutes, encoding="utf-8")

    return {"status": "ok"}
