# Foxtrot-Delta-STT (mp3-opplasting + speed-test)

Denne versjonen lar deg **laste opp en lydfil (f.eks. mp3)** via web-siden *eller* kjøre en **lokal benchmark** fra terminalen. Responsen inneholder timing (sekunder) slik at du kan sammenligne ulike PC-er.

## 1) Kjør som web-app (FastAPI + HTML)

### Installer avhengigheter
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Viktig: ffmpeg
Whisper trenger vanligvis **ffmpeg** for mp3/webm.
- **Windows (winget):** `winget install Gyan.FFmpeg`
- **macOS (brew):** `brew install ffmpeg`
- **Linux (Debian/Ubuntu):** `sudo apt-get install ffmpeg`

### Start API
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Åpne web-klienten
Åpne `client/web/index.html` i nettleseren.
- Velg en mp3 (eller annen lydfil)
- Trykk **"Last opp lydfil"**
- (Valgfritt) huk av **"Lag møtereferat"** for å bruke Ollama/Mistral

**Timing** vises i resultatboksen (`transcribe_seconds`, `audio_duration_seconds`, `real_time_factor`, osv.).

> Tips for sammenligning: Start serveren på nytt for å se "cold start". Første kall inkluderer ofte cache/oppstartseffekter.

## 2) Kjør benchmark fra terminal (uten web)

```bash
python benchmark_stt.py --file ./dinfil.mp3 --model turbo --language no --no-summarize
```

Med møtereferat (krever Ollama kjører lokalt):
```bash
python benchmark_stt.py --file ./dinfil.mp3 --summarize
```

JSON-output (enkelt å logge sammenligninger):
```bash
python benchmark_stt.py --file ./dinfil.mp3 --json
```

## 3) Ollama (valgfritt)
Hvis du skrur på "Lag møtereferat" prøver backend å bruke Ollama på `http://localhost:11434` med modellen `mistral`.
Hvis Ollama ikke kjører, får du fortsatt transkripsjon, og møtereferat-feltet sier at Ollama ikke svarte.
