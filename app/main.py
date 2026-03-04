from __future__ import annotations

import os
import re
import math
import json
import time
import uuid
import pathlib
import textwrap
import subprocess
import datetime
import tempfile
import ctypes
from dataclasses import dataclass
from collections import defaultdict
from typing import Any

import torch
import whisper
import soundfile as sf
from scipy.signal import resample_poly
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
except Exception:
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

try:
    from pyannote.audio import Pipeline  # type: ignore
except Exception:
    Pipeline = None

from app.pipeline.summarize import create_meeting_minutes


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "turbo")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "no")
ASR_BACKEND = os.getenv("ASR_BACKEND", "hf").strip().lower()
HF_WHISPER_MODEL = os.getenv("HF_WHISPER_MODEL", "NbAiLab/nb-whisper-large-distil-turbo-beta").strip()
HF_WHISPER_CHUNK_LENGTH_S = int(os.getenv("HF_WHISPER_CHUNK_LENGTH_S", "30"))
HF_WHISPER_BATCH_SIZE = int(os.getenv("HF_WHISPER_BATCH_SIZE", "8"))

DIARIZATION_ENABLED = os.getenv("DIARIZATION_ENABLED", "1") == "1"
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1")
PYANNOTE_SNAPSHOT_DIR = os.getenv("PYANNOTE_SNAPSHOT_DIR")  # offline snapshot folder
PYANNOTE_MAX_SPEAKERS = int(os.getenv("PYANNOTE_MAX_SPEAKERS", "6"))

TURN_GAP_S = float(os.getenv("TURN_GAP_S", "1.1"))
PARAGRAPH_MAX_CHARS = int(os.getenv("PARAGRAPH_MAX_CHARS", "750"))
WRAP_WIDTH = int(os.getenv("WRAP_WIDTH", "110"))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Torch device: {DEVICE} | CUDA: {torch.cuda.is_available()}")


def _hf_language_name(lang: str) -> str:
    lang = (lang or "").strip().lower()
    mapping = {
        "no": "norwegian",
        "nb": "norwegian",
        "nn": "norwegian",
        "en": "english",
        "sv": "swedish",
        "da": "danish",
    }
    return mapping.get(lang, lang or "norwegian")


class HFWhisperWrapper:
    """Adapter so the rest of the app can keep calling .transcribe(...) like openai-whisper."""

    def __init__(self, model_id_or_path: str, device: str, default_language: str):
        if not model_id_or_path:
            raise RuntimeError("HF_WHISPER_MODEL mangler. Sett sti eller modell-id for Hugging Face-modellen.")
        if AutoModelForSpeechSeq2Seq is None or AutoProcessor is None:
            raise RuntimeError("transformers er ikke installert. Kjør: pip install transformers")

        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        local = pathlib.Path(model_id_or_path)
        model_ref = local if local.exists() else model_id_or_path
        _local_only = os.getenv("HF_HUB_OFFLINE", "1") == "1"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_ref,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=_local_only,
        )
        self.model = model.to(torch.device(device))
        self.processor = AutoProcessor.from_pretrained(model_ref, local_files_only=_local_only)
        self.device = device
        self.torch_dtype = torch_dtype
        self.default_language = default_language

    def transcribe(self, audio_path: str, language: str | None = None, verbose: bool = False) -> dict:
        del verbose  # kept for API compatibility

        lang_name = _hf_language_name(language or self.default_language)
        if not isinstance(audio_path, (str, pathlib.Path)):
            raise RuntimeError("HFWhisperWrapper forventer filsti som input i denne implementasjonen.")

        decoded_path = ensure_wav_for_soundfile(pathlib.Path(audio_path))
        arr, sr = sf.read(str(decoded_path), always_2d=False)
        if getattr(arr, "ndim", 1) > 1:
            arr = arr.mean(axis=1)
        arr = arr.astype("float32", copy=False)
        sr = int(sr)
        target_sr = int(getattr(self.processor.feature_extractor, "sampling_rate", 16000) or 16000)
        if sr != target_sr:
            g = math.gcd(sr, target_sr)
            arr = resample_poly(arr, target_sr // g, sr // g).astype("float32", copy=False)
            sr = target_sr

        chunk_samples = max(1, int(sr * max(1, HF_WHISPER_CHUNK_LENGTH_S)))
        all_text_parts: list[str] = []
        segments: list[dict] = []
        torch_device = torch.device(self.device)

        for start_i in range(0, len(arr), chunk_samples):
            end_i = min(len(arr), start_i + chunk_samples)
            chunk = arr[start_i:end_i]
            if chunk.size == 0:
                continue

            inputs = self.processor(chunk, sampling_rate=sr, return_tensors="pt")
            input_features = inputs["input_features"].to(torch_device)
            if self.device == "cuda":
                input_features = input_features.to(dtype=self.torch_dtype)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_features=input_features,
                    task="transcribe",
                    language=lang_name,
                )

            chunk_text = (self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0] or "").strip()
            if not chunk_text:
                continue

            all_text_parts.append(chunk_text)
            segments.append(
                {
                    "text": chunk_text,
                    "start": round(start_i / sr, 3),
                    "end": round(end_i / sr, 3),
                }
            )

        return {
            "text": " ".join(all_text_parts).strip(),
            "segments": segments,
        }

if ASR_BACKEND == "hf":
    model_ref = HF_WHISPER_MODEL or "NbAiLab/nb-whisper-large-distil-turbo-beta"
    print(f"[INFO] ASR backend: hf | model: {model_ref}")
    whisper_model = HFWhisperWrapper(model_ref, DEVICE, WHISPER_LANGUAGE)
else:
    print(f"[INFO] ASR backend: openai-whisper | model: {WHISPER_MODEL}")
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

        # Viktig: flytt pipeline til GPU hvis mulig, ellers ender den ofte pÃ¥ CPU
        if diar_pipeline is not None and DEVICE == "cuda":
            try:
                diar_pipeline.to(torch.device("cuda"))
                print("[INFO] Moved pyannote pipeline to CUDA")
            except Exception as e:
                print(f"[WARN] Could not move pyannote pipeline to CUDA: {e}")

    except Exception as e:
        diar_pipeline = None
        print(f"[WARN] Diarization disabled (pipeline load failed): {e}")


@dataclass
class MeetingDraft:
    meeting_id: str
    created_at: str
    meeting_name: str
    audio_bytes: bytes
    audio_suffix: str
    transcription: str
    meeting_minutes: str
    speaker_mapping: dict[str, str]
    speakers: list[dict]
    include_minutes: bool
    include_diarization: bool
    timings_seconds: dict[str, float]
    podcast_mode: bool


DRAFTS: dict[str, MeetingDraft] = {}


def ensure_draft(meeting_id: str) -> MeetingDraft:
    d = DRAFTS.get(meeting_id)
    if not d:
        raise HTTPException(status_code=404, detail="Ukjent utkast. Transkriber pÃ¥ nytt.")
    return d


_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_inline(text: str) -> str:
    return _WS_RE.sub(" ", (text or "")).strip()


def wrap_block(text: str, width: int) -> str:
    text = normalize_inline(text)
    if not text:
        return ""
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def format_transcription(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text.replace(". ", ".\n").replace("? ", "?\n").replace("! ", "!\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def is_podcast_like(text: str) -> bool:
    t = (text or "").lower()
    keys = ["podcast", "episode", "i studio", "velkommen til", "du lytter"]
    hits = sum(1 for k in keys if k in t)
    return hits >= 2


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


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


def cleanup_speakers(segments: list[dict], min_total_speech_s: float, keep_top_speakers: int, merge_gap_s: float) -> list[dict]:
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


def ensure_wav_for_soundfile(audio_path: pathlib.Path) -> pathlib.Path:
    audio_path = pathlib.Path(audio_path)

    if audio_path.suffix.lower() in [".wav", ".flac"]:
        return audio_path

    wav_path = audio_path.with_suffix(".wav")

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
        raise RuntimeError("ffmpeg ble ikke funnet i PATH for backend-prosessen.") from e

    if p.returncode != 0 or (not wav_path.exists()) or wav_path.stat().st_size == 0:
        stderr = (p.stderr or "").strip()
        raise RuntimeError(f"Klarte ikke Ã¥ konvertere lyd til WAV. ffmpeg stderr: {stderr[-2000:]}")

    return wav_path


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


def diarize_whole(audio_path: pathlib.Path, max_speakers: int, podcast_mode: bool) -> list[dict]:
    if diar_pipeline is None:
        return []

    audio_path = ensure_wav_for_soundfile(audio_path)
    waveform, sr = load_waveform(audio_path)

    with torch.inference_mode():
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
        min_total = float(os.getenv("PODCAST_MIN_TOTAL_SPEECH_S", "25"))
        keep_top = int(os.getenv("PODCAST_KEEP_TOP_SPEAKERS", "2"))
    else:
        min_total = float(os.getenv("MEETING_MIN_TOTAL_SPEECH_S", "12"))
        keep_top = int(os.getenv("MEETING_KEEP_TOP_SPEAKERS", "8"))

    cleaned = cleanup_speakers(
        raw_segments,
        min_total_speech_s=min_total,
        keep_top_speakers=keep_top,
        merge_gap_s=float(os.getenv("PYANNOTE_MERGE_GAP_S", "0.5")),
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
        best_dist = 1e18
        # velg speaker med hÃ¸yest overlapp (sekunder)
        for ds in diar_segments:
            ds_start = float(ds["start"])
            ds_end = float(ds["end"])
            ov = overlap(ws_start, ws_end, ds_start, ds_end)
            if ov > best_ov:
                best_ov = ov
                best_spk = str(ds["speaker"])
            # fallback: hvis ingen overlapp, velg nÃ¦rmeste diar-segment innenfor en liten terskel
            if ov <= 0.0:
                if ws_end < ds_start:
                    dist = ds_start - ws_end
                elif ds_end < ws_start:
                    dist = ws_start - ds_end
                else:
                    dist = 0.0
                if dist < best_dist:
                    best_dist = dist
                    
        if best_ov <= 0.0 and best_dist <= float(os.getenv("DIAR_NEAREST_TOL_S", "0.25")):
            # finn igjen nÃ¦rmeste og bruk den
            nearest_spk = None
            nearest_dist = 1e18
            for ds in diar_segments:
                ds_start = float(ds["start"])
                ds_end = float(ds["end"])
                if ws_end < ds_start:
                    dist = ds_start - ws_end
                elif ds_end < ws_start:
                    dist = ws_start - ds_end
                else:
                    dist = 0.0
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_spk = str(ds["speaker"])
            if nearest_spk is not None:
                best_spk = nearest_spk

        labeled.append({"speaker": best_spk, "text": normalize_inline(ws.get("text") or ""), "start": ws_start, "end": ws_end})
    return labeled


def merge_to_turns(items: list[dict]) -> list[dict]:
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



def stabilize_short_turns(turns: list[dict]) -> list[dict]:
    """Merge/relable very short speaker turns into neighbors to reduce flicker on overlaps."""
    if not turns:
        return turns
    min_turn_s = float(os.getenv("MIN_TURN_S", "0.6"))
    out = [t.copy() for t in turns]

    for i, t in enumerate(out):
        spk = (t.get("speaker") or "").strip()
        if not spk:
            continue
        dur = float(t.get("end", 0.0) or 0.0) - float(t.get("start", 0.0) or 0.0)
        if dur >= min_turn_s:
            continue
        prev = out[i - 1] if i - 1 >= 0 else None
        nxt = out[i + 1] if i + 1 < len(out) else None
        prev_spk = (prev.get("speaker") or "").strip() if prev else ""
        nxt_spk = (nxt.get("speaker") or "").strip() if nxt else ""
        # Hvis begge naboer er samme speaker, "snap" kort-turnen til den
        if prev_spk and prev_spk == nxt_spk:
            t["speaker"] = prev_spk
    # merge igjen etter relabel
    return merge_to_turns(out)
def render_transcription(items: list[dict]) -> str:
    turns = merge_to_turns(items)
    turns = stabilize_short_turns(turns)

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
    found = set(re.findall(r"\b(Person\s+\d+):", transcription or ""))

    def key(x: str) -> int:
        m = re.search(r"(\d+)$", x)
        return int(m.group(1)) if m else 10**9

    return sorted(found, key=key)


def _sanitize_folder_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "")
    name = name.replace("\n", " ").replace("\r", " ").strip()
    name = re.sub(r"\s+", " ", name)
    return name[:60].strip()


def _short_id(meeting_id: str) -> str:
    return (meeting_id or "")[:8]


def build_folder_name(meeting_id: str, meeting_name: str) -> str:
    stamp = datetime.datetime.now().strftime("%d.%m.%Y")
    sid = _short_id(meeting_id)
    clean = _sanitize_folder_name(meeting_name)
    if clean:
        return f"{stamp}_{clean}_{sid}"
    return f"{stamp}_{sid}"
def format_norwegian_date(dt: datetime.datetime) -> str:
    months = {
        1: "januar",
        2: "februar",
        3: "mars",
        4: "april",
        5: "mai",
        6: "juni",
        7: "juli",
        8: "august",
        9: "september",
        10: "oktober",
        11: "november",
        12: "desember",
    }
    m = months.get(int(dt.month), str(dt.month))
    return f"{dt.day}. {m} {dt.year}"



def get_first_usb_drive_root() -> pathlib.Path:
    if os.name != "nt":
        raise RuntimeError("USB-autodetect er kun implementert for Windows i denne piloten.")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    GetLogicalDrives = kernel32.GetLogicalDrives
    GetDriveTypeW = kernel32.GetDriveTypeW

    DRIVE_REMOVABLE = 2

    bitmask = GetLogicalDrives()
    for i in range(26):
        if not (bitmask & (1 << i)):
            continue
        letter = chr(ord("A") + i)
        root = f"{letter}:\\"
        dtype = GetDriveTypeW(ctypes.c_wchar_p(root))
        if dtype == DRIVE_REMOVABLE:
            p = pathlib.Path(root)
            if p.exists():
                return p

    raise RuntimeError("Fant ingen tilgjengelig minnepenn (removable drive).")


def apply_speaker_mapping(text: str, mapping: dict[str, str]) -> str:
    out = text or ""
    for speaker_id, name in (mapping or {}).items():
        nm = (name or "").strip()
        if nm:
            out = out.replace(f"{speaker_id}:", f"{nm}:")
    return out


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
    mode: str = Form("standard"),
):
    t0_total = time.perf_counter()
    audio_bytes = await file.read()
    suffix = pathlib.Path(file.filename or "").suffix or ".wav"

    meeting_id = str(uuid.uuid4())
    tmp_path = None

    # Lagre audio direkte pÃ¥ minnepennen med Ã©n gang (for fail-safe hvis noe krasjer senere)
    usb_audio_path = None
    try:
        usb_root = get_first_usb_drive_root()
        draft_dir = usb_root / "Foxtrot-Delta-STT" / "meetings" / "_drafts" / meeting_id
        draft_dir.mkdir(parents=True, exist_ok=True)

        # behold original suffix hvis mulig; hvis du alltid vil ha .wav, bruk "audio.wav"
        usb_audio_path = draft_dir / f"audio{suffix}"
        usb_audio_path.write_bytes(audio_bytes)
    except Exception as e:
        # Ikke stopp transkribering om USB mangler/feiler â€” men audio blir da ikke fail-safe pÃ¥ minnepennen
        print(f"[WARN] Klarte ikke lagre audio direkte pÃ¥ USB: {e}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        t0_asr = time.perf_counter()
        result = await run_in_threadpool(
            whisper_model.transcribe,
            tmp_path,
            language=WHISPER_LANGUAGE,
            verbose=False,
        )
        t1_asr = time.perf_counter()

        whisper_segments = result.get("segments", [])
        first_text = " ".join((s.get("text") or "") for s in whisper_segments[:50])
        podcast_mode = is_podcast_like(first_text)

        diar_segments: list[dict] = []
        t0_diar = time.perf_counter()
        if include_diarization:
            try:
                diar_segments = await run_in_threadpool(diarize_whole, pathlib.Path(tmp_path), PYANNOTE_MAX_SPEAKERS, podcast_mode)
            except Exception as e:
                diar_segments = []
                print(f"[WARN] diarization feilet: {e}")
        t1_diar = time.perf_counter()

        labeled_items = label_whisper_segments(whisper_segments, diar_segments)
        transcription = render_transcription(labeled_items)

        minutes = ""
        # mÃ¸te-modus og dato til referat
        meeting_mode = (mode or "standard").strip().lower() or "standard"
        meeting_date_str = format_norwegian_date(datetime.datetime.now())
        t0_sum = time.perf_counter()
        if include_minutes:
            # Frigi GPU-minne slik at Ollama får full VRAM til LLM-en
            if DEVICE == "cuda" and torch.cuda.is_available():
                try:
                    if isinstance(whisper_model, HFWhisperWrapper):
                        whisper_model.model.to("cpu")
                    else:
                        whisper_model.to("cpu")
                except Exception:
                    pass
                try:
                    if diar_pipeline is not None:
                        diar_pipeline.to(torch.device("cpu"))
                except Exception:
                    pass
                torch.cuda.empty_cache()

            minutes = await run_in_threadpool(create_meeting_minutes, transcription, meeting_mode, meeting_date_str)

            # Flytt modellene tilbake til GPU for neste forespørsel
            if DEVICE == "cuda" and torch.cuda.is_available():
                try:
                    if isinstance(whisper_model, HFWhisperWrapper):
                        whisper_model.model.to(torch.device("cuda"))
                    else:
                        whisper_model.to(torch.device("cuda"))
                except Exception:
                    pass
                try:
                    if diar_pipeline is not None:
                        diar_pipeline.to(torch.device("cuda"))
                except Exception:
                    pass
        t1_sum = time.perf_counter()

        speakers_ids = detect_speakers_in_text(transcription)
        speakers = [{"speaker_id": s, "label": s, "name": ""} for s in speakers_ids]

        t1_total = time.perf_counter()

        draft = MeetingDraft(
            meeting_id=meeting_id,
            created_at=datetime.datetime.now().isoformat(),
            meeting_name="",
            audio_bytes=audio_bytes,
            audio_suffix=suffix,
            transcription=transcription,
            meeting_minutes=minutes,
            speaker_mapping={},
            speakers=speakers,
            include_minutes=include_minutes,
            include_diarization=include_diarization,
            podcast_mode=podcast_mode,
            timings_seconds={
                "transcription": round(t1_asr - t0_asr, 3),
                "diarization": round(t1_diar - t0_diar, 3) if include_diarization else 0.0,
                "summarization": round(t1_sum - t0_sum, 3) if include_minutes else 0.0,
                "total": round(t1_total - t0_total, 3),
            },
        )
        DRAFTS[meeting_id] = draft

        return {
            "meeting_id": meeting_id,
            "podcast_mode": podcast_mode,
            "diarization_enabled": diar_pipeline is not None,
            "diarization_requested": include_diarization,
            "minutes_requested": include_minutes,
            "transcription": transcription,
            "meeting_minutes": minutes,
            "timings_seconds": draft.timings_seconds,
            "speakers": speakers,
            "saved": False,
        }

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.put("/drafts/{meeting_id}/name")
async def update_draft_name(meeting_id: str, payload: dict = Body(...)):
    d = ensure_draft(meeting_id)
    d.meeting_name = (payload.get("meeting_name") or payload.get("name") or "").strip()
    return {"meeting_id": meeting_id, "meeting_name": d.meeting_name}


@app.put("/drafts/{meeting_id}/texts")
async def update_draft_texts(meeting_id: str, payload: dict = Body(...)):
    d = ensure_draft(meeting_id)
    d.transcription = payload.get("transcription", d.transcription) or ""
    d.meeting_minutes = payload.get("meeting_minutes", d.meeting_minutes) or ""
    return {"status": "ok"}


@app.put("/drafts/{meeting_id}/speakers")
async def update_draft_speakers(meeting_id: str, mapping: dict[str, str] = Body(...)):
    d = ensure_draft(meeting_id)

    d.speaker_mapping = {k: (v or "").strip() for k, v in (mapping or {}).items()}

    speaker_ids = detect_speakers_in_text(d.transcription)
    d.speakers = [{"speaker_id": s, "label": s, "name": d.speaker_mapping.get(s, "")} for s in speaker_ids]

    d.transcription = apply_speaker_mapping(d.transcription, d.speaker_mapping)
    if d.meeting_minutes:
        d.meeting_minutes = apply_speaker_mapping(d.meeting_minutes, d.speaker_mapping)

    return {
        "speakers": d.speakers,
        "transcription": d.transcription,
        "meeting_minutes": d.meeting_minutes,
    }


@app.post("/drafts/{meeting_id}/save")
async def save_draft(meeting_id: str):
    d = ensure_draft(meeting_id)

    try:
        usb_root = get_first_usb_drive_root()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    folder_name = build_folder_name(d.meeting_id, d.meeting_name)
    out_dir = usb_root / "Foxtrot-Delta-STT" / "meetings" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    final_transcription = apply_speaker_mapping(d.transcription, d.speaker_mapping)
    final_minutes = apply_speaker_mapping(d.meeting_minutes, d.speaker_mapping) if d.meeting_minutes else ""

    (out_dir / "audio.wav").write_bytes(d.audio_bytes)
    (out_dir / "transcription.txt").write_text(final_transcription or "", encoding="utf-8")

    if d.include_minutes:
        (out_dir / "summary.txt").write_text(final_minutes or "", encoding="utf-8")

    meta = {
        "meeting_id": d.meeting_id,
        "created_at": d.created_at,
        "meeting_name": d.meeting_name,
        "include_minutes": d.include_minutes,
        "include_diarization": d.include_diarization,
        "podcast_mode": d.podcast_mode,
        "timings_seconds": d.timings_seconds,
        "speakers": d.speakers,
    }
    (out_dir / "meeting.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    DRAFTS.pop(meeting_id, None)

    return {"status": "saved", "folder": str(out_dir)}

