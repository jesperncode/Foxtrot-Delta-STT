import requests
from requests.exceptions import RequestException

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM = (
    "Du er en profesjonell møteassistent. "
    "Du skal alltid svare på norsk bokmål. "
    "Lag et strukturert møtereferat basert på transkripsjonen. "
    "Ikke skriv på engelsk."
)

def create_meeting_minutes(transcription: str) -> str:
    user_prompt = f"""
Lag et strukturert møtereferat basert på transkripsjonen nedenfor. Den SKAL være på norsk bokmål.

FORMAT:

Tittel
- Gi møtet en kort og passende tittel.

Dato
- Estimer dato hvis den ikke er nevnt.

Deltakere
- List opp deltakere hvis nevnt, ellers skriv "Ikke spesifisert".

Agenda / Tema
- Punktvis liste over hovedtemaene.

Oppsummering av diskusjon
- Kort, tydelig og punktvis.

Beslutninger
- Alle beslutninger som ble tatt.

Oppfølgingspunkter
- Punktliste med ansvarlige hvis mulig.

Neste steg
- Hva skjer videre?

Risikoer / Avklaringer
- Eventuelle uklare punkter eller risikoer.

-------------------------------------------
TRANSKRIPSJON:
{transcription}
-------------------------------------------
""".strip()

    payload = {
        "model": "mistral",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()

        content = (data.get("message", {}) or {}).get("content", "")
        content = content.strip()

        if not content:
            if "error" in data:
                return f"Ollama-feil: {data['error']}"
            return f"Ollama ga tomt svar. Raw: {data}"

        return content

    except RequestException as e:
        return "(Kunne ikke lage møtereferat: Ollama svarte ikke.)\n\n" + str(e)
