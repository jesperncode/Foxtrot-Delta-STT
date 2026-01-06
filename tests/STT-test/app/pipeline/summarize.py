import requests
from requests.exceptions import RequestException

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
        "model": "mixtral:8x7b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except RequestException as e:
        # Hvis Ollama ikke kjører, vil vi ikke at hele /transcribe skal feile når du bare tester STT.
        return (
            "(Kunne ikke lage møtereferat: Ollama/Mistral svarte ikke. "
            "Kjør Ollama lokalt eller fjern avhuking for 'Lag møtereferat'.)\n\n" + str(e)
        )
