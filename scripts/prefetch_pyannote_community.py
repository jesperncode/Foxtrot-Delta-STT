# scripts/prefetch_pyannote_community.py
import os
import sys
from pathlib import Path

import torch
from pyannote.audio import Pipeline

PIPELINE_ID = "pyannote/speaker-diarization-community-1"

def main() -> int:
    hf_home = os.getenv("HF_HOME")
    token = os.getenv("HF_TOKEN")

    if not hf_home:
        print("FEIL: HF_HOME er ikke satt.", file=sys.stderr)
        return 2

    hf_home_path = Path(hf_home).expanduser().resolve()
    hf_home_path.mkdir(parents=True, exist_ok=True)

    if not token:
        print("FEIL: HF_TOKEN er ikke satt. Sett den midlertidig i terminalen.", file=sys.stderr)
        return 3

    print("Python:", sys.executable)
    print("HF_HOME:", str(hf_home_path))
    print("Pipeline:", PIPELINE_ID)

    pipeline = Pipeline.from_pretrained(
        PIPELINE_ID,
        token=token,
    )

    pipeline.to(torch.device("cpu"))
    print("OK: Pipeline lastet og cachet lokalt.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
