import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def create_meeting_minutes(transcription: str) -> str:
    prompt = f"""
Du er en profesjonell møteassistent. Lag et strukturert møtereferat basert på transkripsjonen nedenfor. Den SKAL være på norsk bokmål

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

Lag referatet profesjonelt og strukturert.
"""

    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()

    return data["response"]
