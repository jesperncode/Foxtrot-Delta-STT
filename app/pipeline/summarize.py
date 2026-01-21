from __future__ import annotations

import os
import requests
from typing import List


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"

MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")

SYSTEM_RULES = (
    "KRAV: Svar må være 100% norsk bokmål. Ikke bruk engelsk. "
    "Bruk kun informasjon som finnes i teksten. Ikke finn på."
)


def _ollama_generate(prompt: str, temperature: float = 0.2, timeout_s: int = 300) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout_s)
    if r.status_code == 404:
        raise RuntimeError(
            f"Fikk 404 fra Ollama på {OLLAMA_GENERATE_URL}. "
            f"Sjekk at OLLAMA_HOST peker riktig."
        )

    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def _chunk_text_by_lines(text: str, max_chars: int = 12_000) -> List[str]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    size = 0

    for ln in lines:
        if len(ln) > max_chars:
            ln = ln[:max_chars]

        if size + len(ln) + 1 > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0

        buf.append(ln)
        size += len(ln) + 1

    if buf:
        chunks.append("\n".join(buf))

    return chunks


def create_meeting_minutes(transcription: str) -> str:
    transcription = (transcription or "").strip()
    if not transcription:
        return (
            "Tittel\n- Ikke spesifisert.\n\n"
            "Dato\n- Ikke spesifisert.\n\n"
            "Deltakere\n- Ikke spesifisert.\n\n"
            "Agenda / Tema\n- Ikke spesifisert.\n\n"
            "Oppsummering av diskusjon\n- Ikke spesifisert.\n\n"
            "Beslutninger\n- Ikke spesifisert.\n\n"
            "Oppfølgingspunkter\n- Ikke spesifisert.\n\n"
            "Neste steg\n- Ikke spesifisert.\n\n"
            "Risikoer / Avklaringer\n- Ikke spesifisert.\n"
        )

    chunks = _chunk_text_by_lines(transcription, max_chars=12_000)

    notes: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        prompt = f"""{SYSTEM_RULES}

Du får DEL {i}/{len(chunks)} av en transkripsjon. Transkripsjonen kan inneholde linjer som starter med "Person N:".

Oppgave:
- Lag KORTE, presise møtenotater fra denne delen.
- Punktvis, tydelig, kun fakta fra teksten.

Returner i denne strukturen (nøyaktig):
Temaer:
- ...

Beslutninger:
- ...

Oppfølgingspunkter (med ansvarlig hvis nevnt):
- ...

Risikoer/Avklaringer:
- ...

TEKST:
{ch}
"""
        notes.append(_ollama_generate(prompt, temperature=0.2))

    combined_notes = "\n\n".join(
        f"DELNOTATER {idx}:\n{txt}" for idx, txt in enumerate(notes, start=1)
    )

    final_prompt = f"""{SYSTEM_RULES}

Lag ett samlet, profesjonelt møtereferat basert på delnotatene.

FORMAT (nøyaktig med disse overskriftene):
Tittel
- ...

Dato
- ...

Deltakere
- ...

Agenda / Tema
- ...

Oppsummering av diskusjon
- ...

Beslutninger
- ...

Oppfølgingspunkter
- ...

Neste steg
- ...

Risikoer / Avklaringer
- ...

Krav:
- 100% norsk bokmål
- Ikke finn på ting
- Hvis noe mangler: skriv "Ikke spesifisert"
- Ikke referer til "delnotater" i svaret
- Ikke skriv på engelsk

DELNOTATER:
{combined_notes}
"""
    return _ollama_generate(final_prompt, temperature=0.2)
