import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def create_meeting_minutes(text: str) -> str:
    """
    Genererer møtereferat basert på transkribert tekst.
    Bruker lokal Mistral via Ollama
    """

    payload = {
        "model": "mistral",
        "prompt": f"Oppsummer det følgende møtet på en kort, tydelig og strukturert måte:\n\n{text}"
    }

    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()

    return data["response"]
