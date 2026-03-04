from __future__ import annotations

import json
import argparse
import pathlib
from collections import defaultdict

import torch
import soundfile as sf
from pyannote.audio import Pipeline


def load_waveform(audio_path: pathlib.Path) -> tuple[torch.Tensor, int]:
    wav, sr = sf.read(str(audio_path), always_2d=False)
    if wav.ndim == 1:
        wav = wav[None, :]
    else:
        wav = wav.T
    return torch.from_numpy(wav).float(), int(sr)


def ensure_annotation(diar):
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


def diarize_whole(pipe: Pipeline, audio_path: pathlib.Path, max_speakers: int, min_total: float, keep_top: int, merge_gap: float):
    waveform, sr = load_waveform(audio_path)

    diar_out = pipe({"waveform": waveform, "sample_rate": sr}, min_speakers=1, max_speakers=max_speakers)
    diar = ensure_annotation(diar_out)

    segments = []
    for seg, _, lbl in diar.itertracks(yield_label=True):
        segments.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(lbl)})

    segments = cleanup_speakers(segments, min_total_speech_s=min_total, keep_top_speakers=keep_top, merge_gap_s=merge_gap)

    speaker_map: dict[str, str] = {}
    counter = 1
    for s in segments:
        raw = s["speaker"]
        if raw not in speaker_map:
            speaker_map[raw] = f"Person {counter}"
            counter += 1
        s["speaker"] = speaker_map[raw]

    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    ap.add_argument("--max-speakers", type=int, default=6)
    ap.add_argument("--merge-gap", type=float, default=0.4)

    ap.add_argument("--min-total-speech", type=float, default=12.0)
    ap.add_argument("--keep-top", type=int, default=8)

    ap.add_argument("--podcast", action="store_true")
    ap.add_argument("--podcast-min-total", type=float, default=25.0)
    ap.add_argument("--podcast-keep-top", type=int, default=2)

    args = ap.parse_args()

    audio_path = pathlib.Path(args.audio).resolve()
    pipe = Pipeline.from_pretrained(args.model)

    if args.podcast:
        min_total = args.podcast_min_total
        keep_top = args.podcast_keep_top
    else:
        min_total = args.min_total_speech
        keep_top = args.keep_top

    segments = diarize_whole(
        pipe,
        audio_path,
        max_speakers=args.max_speakers,
        min_total=min_total,
        keep_top=keep_top,
        merge_gap=args.merge_gap,
    )

    print(json.dumps(segments, ensure_ascii=False))


if __name__ == "__main__":
    main()
