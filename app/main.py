# app/main.py
from __future__ import annotations

import os
import pathlib
import re
import time
import textwrap
import subprocess
from collections import defaultdict
from typing import Any

import torch
import whisper
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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

# ---- FORMATERINGSKNOTTER ----
TURN_GAP_S = float(os.getenv("TURN_GAP_S", "1.1"))  # pause > dette -> nytt avsnitt/turn
PARAGRAPH_MAX_CHARS = int(os.getenv("PARAGRAPH_MAX_CHARS", "750"))
WRAP_WIDTH = int(os.getenv("WRAP_WIDTH", "110"))  # linjebredde i textarea

# ---- DEBUG ----
GPU_DEBUG = os.getenv("GPU_DEBUG", "0") == "1"

# ---- SILENCE CHECK (valgfritt men anbefalt) ----
REJECT_SILENT_AUDIO = os.getenv("REJECT_SILENT_AUDIO", "1") == "1"
SILENCE_MAX_VOLUME_DB_THRESHOLD = float(os.getenv("SILENCE_MAX_VOLUME_DB_THRESHOLD", "-55.0"))


# ==========================================================
# DEVICE + MODELLER (last én gang)
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Torch device: {DEVICE}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


def _gpu_mem(tag: str) -> None:
    if not GPU_DEBUG:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() // 1024**2
        resv = torch.cuda.memory_reserved() // 1024**2
        print(f"[GPU] {tag}: allocated={alloc}MB reserved={resv}MB")


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

        # >>> VIKTIG: flytt pyannote til GPU (ellers ender den ofte på CPU)
        if diar_pipeline is not None and DEVICE == "cuda":
            try:
                diar_pipeline.to(torch.device("cuda"))
                print("[INFO] Moved pyannote pipeline to CUDA")
                _gpu_mem("after pyannote.to(cuda)")
            except Exception as e:
                print(f"[WARN] Could not move pyannote pipeline to CUDA: {e}")

    except Exception as e:
        diar_pipeline = None
        print(f"[WARN] Diarization disabled (pipeline load failed): {e}")


# ==========================================================
# MEETING-ID -> DIR cache
# ==========================================================
MEETING_DIRS: dict[str, pathlib.Path] = {}  # fylles i /transcribe


def get_meeting_dir(meeting_id: str) -> pathlib.Path:
    d = MEETING_DIRS.get(meeting_id)
    if d and d.exists():
        return d
    raise HTTPException(
        status_code=404,
        detail="Ukjent meeting_id (server restartet eller møtemappe ikke funnet). Kjør /transcribe på nytt.",
    )


# ==========================================================
# TEKSTNORMALISERING
# ==========================================================
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_inline(text: str) -> str:
    """Gjør all whitespace (inkl. \n) om til enkel mellomrom."""
    return _WS_RE.sub(" ", (text or "")).strip()


def is_podcast_like(text: str) -> bool:
    t = (text or "").lower()
    keys = ["podcast", "episode", "i studio", "velkommen til", "du lytter"]
    hits = sum(1 for k in keys if k in t)
    return hits >= 2


def wrap_block(text: str, width: int) -> str:
    """
    Linjebryting for lesbarhet i textarea.
    Splitter ikke ord.
    """
    text = normalize_inline(text)
    if not text:
        return ""
    return textwrap.fill(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


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


def ensure_wav_for_soundfile(audio_path: pathlib.Path) -> pathlib.Path:
    """
    libsndfile/soundfile støtter ofte ikke .webm/.m4a/.mp3 på Windows.
    Vi konverterer derfor til mono 16k WAV før diarization/waveform-lesing.
    """
    audio_path = pathlib.Path(audio_path)

    if audio_path.suffix.lower() in [".wav", ".flac"]:
        return audio_path

    wav_path = audio_path.with_suffix(".wav")

    # Rebruk hvis eksisterer og er nyere/lik
    if wav_path.exists():
        try:
            if wav_path.stat().st_mtime >= audio_path.stat().st_mtime and wav_path.stat().st_size > 0:
                return wav_path
        except OSError:
            pass

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]

    try:
        p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg ble ikke funnet i PATH for backend-prosessen. Installer ffmpeg eller legg ffmpeg/bin i PATH."
        ) from e

    if p.returncode != 0 or (not wav_path.exists()) or wav_path.stat().st_size == 0:
        stderr = (p.stderr or "").strip()
        raise RuntimeError(
            "Klarte ikke å konvertere lyd til WAV med ffmpeg. "
            f"Input: {audio_path}. ffmpeg stderr: {stderr[-2000:]}"
        )

    return wav_path


_VOL_MEAN_RE = re.compile(r"mean_volume:\s*(-?inf|[-\d\.]+)\s*dB")
_VOL_MAX_RE = re.compile(r"max_volume:\s*(-?inf|[-\d\.]+)\s*dB")


def ffmpeg_volumedetect(audio_path: pathlib.Path) -> dict[str, float | str]:
    """
    Returnerer {"mean_volume_db": float|str, "max_volume_db": float|str}
    -inf betyr helt null.
    """
    cmd = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "NUL" if os.name == "nt" else "/dev/null",
    ]
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    txt = (p.stderr or "") + "\n" + (p.stdout or "")

    m_mean = _VOL_MEAN_RE.search(txt)
    m_max = _VOL_MAX_RE.search(txt)

    def parse(v: str) -> float | str:
        if v in ("-inf", "inf"):
            return v
        try:
            return float(v)
        except Exception:
            return v

    out: dict[str, float | str] = {}
    if m_mean:
        out["mean_volume_db"] = parse(m_mean.group(1))
    if m_max:
        out["max_volume_db"] = parse(m_max.group(1))
    return out


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

    # >>> FIX: alltid WAV før soundfile
    audio_path = ensure_wav_for_soundfile(audio_path)

    waveform, sr = load_waveform(audio_path)

    if DEVICE == "cuda":
        _gpu_mem("before diarization call")

    with torch.inference_mode():
        diar_out = diar_pipeline(
            {"waveform": waveform, "sample_rate": sr},
            min_speakers=1,
            max_speakers=max_speakers,
        )

    if DEVICE == "cuda":
        _gpu_mem("after diarization call")

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


# ==========================================================
# LABEL + TURNING
# ==========================================================
def label_whisper_segments(whisper_segments: list[dict], diar_segments: list[dict]) -> list[dict]:
    """
    Returnerer items med timing:
      - med diarization: {"speaker": "...", "text": "...", "start": float, "end": float}
      - uten diarization: {"text": "...", "start": float, "end": float}
    """
    if not diar_segments:
        out = []
        for ws in whisper_segments:
            out.append(
                {
                    "text": normalize_inline(ws.get("text") or ""),
                    "start": float(ws.get("start", 0.0) or 0.0),
                    "end": float(ws.get("end", 0.0) or 0.0),
                }
            )
        return out

    labeled: list[dict] = []
    for ws in whisper_segments:
        ws_start = float(ws.get("start", 0.0) or 0.0)
        ws_end = float(ws.get("end", 0.0) or 0.0)

        best_spk = "Ukjent"
        best_ov = 0.0
        for ds in diar_segments:
            ov = overlap(ws_start, ws_end, float(ds["start"]), float(ds["end"]))
            if ov > best_ov:
                best_ov = ov
                best_spk = str(ds["speaker"])

        labeled.append(
            {
                "speaker": best_spk,
                "text": normalize_inline(ws.get("text") or ""),
                "start": ws_start,
                "end": ws_end,
            }
        )
    return labeled


def merge_to_turns(items: list[dict]) -> list[dict]:
    """
    Lager "turns"/avsnitt basert på:
      - speaker-skifte (hvis diarization)
      - pause > TURN_GAP_S
      - eller hvis avsnitt blir for langt (PARAGRAPH_MAX_CHARS)
    """
    out: list[dict] = []
    for it in items:
        txt = normalize_inline(it.get("text") or "")
        if not txt:
            continue

        spk = (it.get("speaker") or "").strip()
        st = float(it.get("start", 0.0) or 0.0)
        en = float(it.get("end", 0.0) or 0.0)

        if not out:
            base = {"text": txt, "start": st, "end": en}
            if spk:
                base["speaker"] = spk
            out.append(base)
            continue

        prev = out[-1]
        prev_spk = (prev.get("speaker") or "").strip()
        prev_end = float(prev.get("end", 0.0) or 0.0)

        gap = st - prev_end

        same_speaker = (spk and prev_spk == spk) or (not spk and "speaker" not in prev)

        too_long = len(prev.get("text", "")) >= PARAGRAPH_MAX_CHARS
        new_paragraph = (gap > TURN_GAP_S) or too_long or (not same_speaker)

        if not new_paragraph:
            prev["text"] = normalize_inline(prev.get("text", "") + " " + txt)
            prev["end"] = max(prev_end, en)
        else:
            base = {"text": txt, "start": st, "end": en}
            if spk:
                base["speaker"] = spk
            out.append(base)

    return out


def render_transcription(items: list[dict]) -> str:
    turns = merge_to_turns(items)

    blocks: list[str] = []
    for t in turns:
        spk = (t.get("speaker") or "").strip()
        txt = t.get("text") or ""
        wrapped = wrap_block(txt, WRAP_WIDTH)
        if not wrapped:
            continue

        if spk:
            blocks.append(f"{spk}: {wrapped}")
        else:
            blocks.append(wrapped)

    return "\n\n".join(blocks).strip()


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
async def transcribe(
    file: UploadFile = File(...),
    include_minutes: bool = Form(False),
    include_diarization: bool = Form(False),
):
    t0_total = time.perf_counter()
    content = await file.read()

    try:
        meeting_id, meeting_dir = create_meeting()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    MEETING_DIRS[meeting_id] = pathlib.Path(meeting_dir)

    suffix = pathlib.Path(file.filename).suffix or ".wav"
    audio_path = meeting_dir / f"audio{suffix}"
    audio_path.write_bytes(content)

    # (Valgfritt) stopp tidlig hvis opptaket er stille
    if REJECT_SILENT_AUDIO:
        try:
            vd = ffmpeg_volumedetect(audio_path)
            maxv = vd.get("max_volume_db", None)
            if maxv == "-inf":
                raise HTTPException(
                    status_code=400,
                    detail="Opptaket ser ut til å være helt stille (max_volume=-inf). Sjekk riktig mikrofon og at den ikke var muted.",
                )
            if isinstance(maxv, float) and maxv < SILENCE_MAX_VOLUME_DB_THRESHOLD:
                raise HTTPException(
                    status_code=400,
                    detail=f"Opptaket har ekstremt lavt nivå (max_volume={maxv} dB). Sjekk input/gain/mute.",
                )
        except HTTPException:
            raise
        except Exception as e:
            print(f"[WARN] volumedetect failed: {e}")

    # Whisper (alltid)
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
    diar_segments: list[dict] = []
    t0_diar = time.perf_counter()
    if include_diarization:
        diar_segments = await run_in_threadpool(
            diarize_whole,
            audio_path,
            PYANNOTE_MAX_SPEAKERS,
            podcast_mode,
        )
    t1_diar = time.perf_counter()

    labeled_items = label_whisper_segments(whisper_segments, diar_segments)
    transcription = render_transcription(labeled_items)

    # Møtereferat (valgfritt)
    minutes = ""
    t0_sum = time.perf_counter()
    if include_minutes:
        minutes = await run_in_threadpool(create_meeting_minutes, transcription)
    t1_sum = time.perf_counter()

    (meeting_dir / "transcription.txt").write_text(transcription, encoding="utf-8")
    if include_minutes:
        (meeting_dir / "summary.txt").write_text(minutes, encoding="utf-8")

    t1_total = time.perf_counter()

    speakers = detect_speakers_in_text(transcription)
    speaker_objs = [{"speaker_id": s, "label": s, "name": ""} for s in speakers]

    return {
        "meeting_id": meeting_id,
        "podcast_mode": podcast_mode,
        "diarization_enabled": diar_pipeline is not None,
        "diarization_requested": include_diarization,
        "minutes_requested": include_minutes,
        "transcription": transcription,
        "meeting_minutes": minutes,
        "timings_seconds": {
            "transcription": round(t1_asr - t0_asr, 3),
            "diarization": round(t1_diar - t0_diar, 3) if include_diarization else 0.0,
            "summarization": round(t1_sum - t0_sum, 3) if include_minutes else 0.0,
            "total": round(t1_total - t0_total, 3),
        },
        "speakers": speaker_objs,
    }


@app.put("/meetings/{meeting_id}/speakers")
async def update_speakers(meeting_id: str, mapping: dict[str, str]):
    meeting_dir = get_meeting_dir(meeting_id)

    t_path = meeting_dir / "transcription.txt"
    s_path = meeting_dir / "summary.txt"

    if not t_path.exists():
        raise HTTPException(status_code=404, detail="Mangler transkripsjon for dette møtet")

    transcription = t_path.read_text(encoding="utf-8")
    minutes = s_path.read_text(encoding="utf-8") if s_path.exists() else ""

    transcription2 = replace_speakers(transcription, mapping)
    minutes2 = replace_speakers(minutes, mapping) if minutes else ""

    t_path.write_text(transcription2, encoding="utf-8")
    if s_path.exists():
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

    if isinstance(minutes, str) and minutes.strip():
        (meeting_dir / "summary.txt").write_text(minutes, encoding="utf-8")

    return {"status": "ok"}
