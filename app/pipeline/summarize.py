from __future__ import annotations

import os
import re
from typing import List

import requests

# Fjerner <think>...</think>-blokker som noen modeller (f.eks. DeepSeek) inkluderer i svaret
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ----------------------------
# Ollama config
# ----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"

# Hvilken LLM-modell som brukes til referatgenerering
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:32b-instruct-q4_K_M")

# Default timeouts (seconds)
# HEAVY brukes til kall med mer innhold, f.eks. final-pass og komprimering
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "600"))         # Per-chunk notes (32b kan bruke tid)
OLLAMA_TIMEOUT_S_HEAVY = int(os.getenv("OLLAMA_TIMEOUT_S_HEAVY", "1200"))  # Finalgenerering og komprimering

# Context window — qwen2.5:7b støtter 32K, llama3:8b maks 8K
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))

# Chunking — maks tegn per del ved splitting av lange transkripsjoner
SUMMARY_CHUNK_MAX_CHARS = int(os.getenv("SUMMARY_CHUNK_MAX_CHARS", "40000"))
# Hopp over notes-pass og send transkripsjon direkte til final hvis den passer i én chunk
ENABLE_DIRECT_FINAL_SINGLE_CHUNK = os.getenv("ENABLE_DIRECT_FINAL_SINGLE_CHUNK", "0").strip() == "1"

# Token caps — uten think-modell kan vi sette disse høyere
NOTES_NUM_PREDICT = int(os.getenv("NOTES_NUM_PREDICT", "6000"))
FINAL_NUM_PREDICT = int(os.getenv("FINAL_NUM_PREDICT", "-1"))  # -1 = ubegrenset

# Forhåndskorriger transkripsjonen via LLM før oppsummering (som regel deaktivert — øker latens)
ENABLE_TRANSCRIPTION_NORMALIZATION = os.getenv("ENABLE_TRANSCRIPTION_NORMALIZATION", "0").strip() == "1"

# Komprimer delnotater til én kompakt blokk før final-pass — reduserer prompt-størrelse kraftig
ENABLE_NOTES_COMPRESSION = os.getenv("ENABLE_NOTES_COMPRESSION", "1").strip() == "1"
# Hard grense for hvor mye notattekst som sendes inn til komprimeringen
NOTES_COMPRESSION_MAX_CHARS = int(os.getenv("NOTES_COMPRESSION_MAX_CHARS", "18000"))

# Systeminstruksjon som alltid sendes med til modellen — definerer språk, stil og innholdskrav
SYSTEM_RULES = (
    "DU ER EN PROFESJONELL MØTEREFERENT FOR NORSKE VIRKSOMHETER.\n"
    "KRITISK: Svar KUN på norsk bokmål. Aldri bruk engelsk eller andre språk.\n\n"
    "Språkkrav:\n"
    "- 100 % norsk bokmål (Norge).\n"
    "- Ikke bruk danske eller svenske ordformer.\n"
    "- Bruk offisiell norsk rettskrivning.\n\n"
    "Forkortelser og egennavn:\n"
    "- Behold ALLE forkortelser nøyaktig slik de står.\n"
    "- Ikke utvid, forklar eller oversett forkortelser.\n"
    "- Behold egennavn nøyaktig slik de står.\n\n"
    "Innholdskrav:\n"
    "- Bruk kun informasjon som eksplisitt finnes i teksten.\n"
    "- Ikke tolk intensjoner som ikke er sagt.\n"
    "- Ikke finn på beslutninger eller fakta.\n"
    "- Hvis noe ikke fremgår: skriv 'Ikke spesifisert'.\n"
    "- ALDRI skriv at noe ble 'besluttet' eller 'bestemt' med mindre "
    "transkripsjonen eksplisitt bekrefter at en formell beslutning ble fattet. "
    "Hvis usikkert: skriv 'Ikke besluttet' eller 'Ikke spesifisert'.\n\n"
    "REFERATKVALITET – KRITISK:\n"
    "Referatet skal være så komplett og selvstendig at leseren aldri trenger å gå tilbake "
    "til lydopptaket eller transkripsjonen. Dette krever:\n\n"
    "1. KONTEKST:\n"
    "- Forklar alltid hvorfor en sak er oppe — hva er bakgrunnen og hva står på spill.\n"
    "- Inkluder relevante tall, datoer, lovhenvisninger og tidligere beslutninger som nevnes.\n"
    "- Hvis eksterne parter, regler eller rammebetingelser påvirker saken: forklar hvordan.\n\n"
    "2. DYNAMIKK OG ARGUMENTASJON:\n"
    "- Gjengi hvilke synspunkter og argumenter som kom frem — ikke bare at 'det ble diskutert'.\n"
    "- Hvis det var uenighet: beskriv hvilke posisjoner som stod mot hverandre og hvorfor.\n"
    "- Hvis talere er identifisert i transkripsjonen: noter hvem som hadde hvilke synspunkter.\n"
    "- Hvis talere ikke er identifisert: beskriv posisjonene uten å tilskrive dem til navngitte personer.\n"
    "- Ikke flat ut debatter til nøytrale beskrivelser — bevar det som faktisk skjedde.\n\n"
    "3. UTFALL OG KONSEKVENSER:\n"
    "- Skill alltid strengt mellom diskutert, foreslått og vedtatt.\n"
    "- Ved avstemninger: oppgi stemmetall og forklar den praktiske konsekvensen av resultatet.\n"
    "- Hvis regler eller prosedyrer avgjorde utfallet: forklar hva regelen sier og hvilken "
    "betydning den fikk.\n"
    "- Skriv aldri at noe 'ble besluttet' hvis det kun ble diskutert.\n\n"
    "4. OPPFØLGING:\n"
    "- Noter hvem som har ansvar for hva etter møtet.\n"
    "- Inkluder frister hvis de nevnes.\n"
    "- Løft frem uavklarte spørsmål og åpne punkter eksplisitt.\n\n"
    "Kvalitetskrav:\n"
    "- Skriv profesjonelt og presist.\n"
    "- Ingen fyllord eller gjentakelser.\n"
    "- Ingen direkte sitater.\n"
    "- Ikke skriv 'Det ble diskutert om X' — skriv hva som faktisk ble sagt og argumentert.\n"
    "- Ikke gjenta samme informasjon i flere seksjoner.\n\n"
    "Dekkingskrav:\n"
    "- Alle temaer, argumenter, tall, navn og beslutninger skal være med.\n"
    "- Ikke avslutt for tidlig. Skriv til alt er dekket.\n"
)


# Modus-spesifikke prompter for notes-pass (chunking) og final-pass.
# Hvert modus har sin egen notat-instruksjon og sluttinstruksjon.
MODE_PROMPTS = {
    # Generell møtesammenfatning uten fast struktur — modellen velger selv hensiktsmessig format
    "standard": {
        "notes": (
            "Identifiser maksimalt 5-6 distinkte hovedtemaer. "
            "Slå sammen beslektet innhold under samme tema fremfor å lage mange små. "
            "For hvert tema: noter bakgrunn, konkrete argumenter fra ulike parter, "
            "tall og lovhenvisninger nøyaktig som nevnt, og hva som faktisk ble besluttet eller ikke. "
            "Ikke dupliser informasjon på tvers av temaer."
        ),
        "final": (
            "Lag en strukturert møtesammenfatning der du selv velger den mest hensiktsmessige strukturen "
            "basert på innholdet, ikke et fast skjema."
        ),
    },
    # Formelt møtereferat med tydelig skille mellom diskusjon, beslutninger og oppfølging
    "møtereferat": {
        "notes": (
            "Identifiser hovedtemaer, konkrete diskusjonspunkter, eventuelle beslutninger, "
            "ansvar og åpne spørsmål. Skill tydelig mellom hva som er diskutert og hva som er besluttet."
        ),
        "final": (
            "Lag et strukturert og profesjonelt møtereferat som gir full oversikt over "
            "hva som ble diskutert, hva som ble besluttet, og hva som krever oppfølging."
        ),
    },
    # Militært referat med sak-for-sak-struktur: fakta → vurdering → alternativer → beslutning
    "mil": {
        "notes": (
            "Trekk ut hovedsaker og beslutningspunkter i en strukturert, disiplinert form. "
            "For hver sak: fakta/situasjon, vurdering/risiko, handlingsalternativer, anbefaling, beslutning og tiltak."
        ),
        "final": (
            "Lag et strukturert referat etter militær modell (formål/agenda/rammer, sak-for-sak med fakta–vurdering–alternativer–anbefaling–beslutning, og avslutning med tiltaksliste)."
        ),
    },
}


# ----------------------------
# Core helpers
# ----------------------------
def _ollama_generate(prompt: str, temperature: float = 0.2, timeout_s: int | None = None, num_predict: int | None = None) -> str:
    """Sender en prompt til Ollama og returnerer svaret renset for think-blokker."""
    options: dict = {"temperature": float(temperature), "num_ctx": OLLAMA_NUM_CTX}
    if num_predict is not None:
        options["num_predict"] = int(num_predict)
    payload = {
        "model": MODEL,
        # Prompten avsluttes alltid med en eksplisitt norsk-påminnelse
        "prompt": prompt.rstrip() + "\n\nSvar KUN på norsk bokmål:",
        "system": SYSTEM_RULES,
        "stream": False,
        "options": options,
    }
    timeout_s = int(timeout_s or OLLAMA_TIMEOUT_S)

    r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout_s)
    if r.status_code == 404:
        raise RuntimeError(
            f"Fikk 404 fra Ollama på {OLLAMA_GENERATE_URL}. "
            f"Sjekk at OLLAMA_HOST peker riktig."
        )
    r.raise_for_status()
    data = r.json()
    raw = (data.get("response") or "").strip()
    # Fjern eventuelle <think>-blokker fra svaret før det returneres
    return _THINK_RE.sub("", raw).strip()


def _chunk_text_by_lines(text: str, max_chars: int) -> List[str]:
    """Deler tekst i biter på maks max_chars tegn, uten å kutte midt i en linje."""
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    size = 0

    for ln in lines:
        # Klipp enkeltlinjer som alene overskrider grensen
        if len(ln) > max_chars:
            ln = ln[:max_chars]

        # Flush buffer når neste linje ville sprenge grensen
        if size + len(ln) + 1 > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0

        buf.append(ln)
        size += len(ln) + 1

    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _postprocess_norwegian_spelling(text: str) -> str:
    """
    Minimal og deterministisk etterkorrigering av noen få vanlige feilformer.
    VIKTIG: dette er bevisst "lite", for ikke å ødelegge egennavn/forkortelser.
    """
    if not text:
        return text

    corrections = {
        "Rusland": "Russland",
        "hedder": "heter",
        "volym": "volum",
        "brand": "brann",
        "brandvern": "brannvern",
        "litimion": "litiumion",
        "lithiumion": "litiumion",
        "mulighed": "mulighet",
        "muligheden": "muligheten",
        "organisation": "organisasjon",
        "organisations": "organisasjons",
    }
    for src, dst in corrections.items():
        text = text.replace(src, dst)
    return text


def _normalize_transcription_no_semantic_change(text: str) -> str:
    """
    Retter åpenbare skrivefeil og grammatikk til norsk bokmål uten å endre meningsinnhold.
    Bevarer forkortelser, egennavn, tall, datoer og tekniske betegnelser.
    """
    if not text:
        return text

    prompt = f"""
Oppgave:
- Rett åpenbare skrivefeil og grammatikk til korrekt norsk bokmål.
- IKKE endre meningsinnhold.
- IKKE legg til eller fjern informasjon.
- IKKE omskriv mer enn nødvendig.
- Behold alle forkortelser og egennavn nøyaktig som i originalen.
- Behold tall, datoer og tekniske betegnelser.

Returner kun den korrigerte teksten, uten forklaring.

TEKST:
{text}
"""
    corrected = _ollama_generate(prompt, temperature=0.0, timeout_s=OLLAMA_TIMEOUT_S_HEAVY)
    return _postprocess_norwegian_spelling(corrected)


def _compress_notes_for_final(notes_text: str, mode_key: str) -> str:
    """
    Komprimerer delnotater før final-pass (reduserer prompt-størrelse kraftig, særlig for MIL/OKONOMI/OPPDRAG).
    """
    if not notes_text:
        return notes_text

    # hard cap: ikke send enorme mengder inn i komprimeringen
    if len(notes_text) > NOTES_COMPRESSION_MAX_CHARS:
        notes_text = notes_text[:NOTES_COMPRESSION_MAX_CHARS]

    prompt = f"""
Oppgave:
Du skal komprimere DELNOTATER til en kortere, men fortsatt dekkende notatpakke for et FINAL-referat.

Krav:
- Ikke legg til nye fakta.
- Ikke fjern viktige tall, datoer, beslutninger, ansvar, frister, risiko eller avklaringer.
- Slå sammen repetisjon.
- Behold forkortelser og egennavn nøyaktig som de står.
- Ikke bruk sitater.
- Ikke referer til at dette er delnotater.

Returner nøyaktig denne strukturen:

Temaer:
- ...

Diskusjon:
- ...

Forslag:
- ...

Beslutninger:
- ...

Oppfølgingspunkter (med ansvarlig og frist hvis nevnt):
- ...

Risikoer/Avklaringer:
- ...

Modus: {mode_key}

DELNOTATER:
{notes_text}
"""
    np = NOTES_NUM_PREDICT
    compressed = _ollama_generate(prompt, temperature=0.0, timeout_s=OLLAMA_TIMEOUT_S_HEAVY, num_predict=np)
    return _postprocess_norwegian_spelling(compressed)


def _length_requirements(transcription: str, mode: str) -> str:
    """Returnerer en lengdeinstruksjon til modellen basert på transkripsjonens ordtall og modus.
    Lengre innhold krever et mer detaljert referat — sikrer at modellen ikke kutter for tidlig."""
    words = len((transcription or "").split())

    if mode == "ir":
        if words <= 700:
            return "Du MÅ skrive minst 500 ord. Dekk alle temaer grundig – ikke avslutt for tidlig."
        if words <= 1800:
            return "Du MÅ skrive minst 900 ord. Hvert tema skal ha sitt eget avsnitt med alle detaljer."
        if words <= 4000:
            return "Du MÅ skrive minst 1400 ord. Hvert tema skal ha sitt eget avsnitt. Ingen poenger skal kuttes."
        return "Du MÅ skrive minst 2200 ord. Hvert tema skal ha sitt eget avsnitt med alle detaljer og nyanser."

    if words <= 500:
        return "Dekk alle temaer grundig med konkrete fakta, argumenter og konklusjon per tema. Ikke avslutt før alt er dekket."
    if words <= 1200:
        return "Hvert tema skal ha sitt eget avsnitt med konkrete detaljer, argumenter og hva som ble konkludert. Ikke kutt noe. Ikke gjenta deg selv."
    if words <= 2500:
        return "Hvert tema skal ha sin egen seksjon. Beskriv presist hva som ble sagt, hvilke argumenter som kom frem og hva som ble konkludert. Ikke gjenta informasjon fra andre seksjoner."
    if words <= 5000:
        return "Hvert tema som er diskutert skal ha sin egen seksjon med fullstendig og nøyaktig dekning. Ingen saker skal kuttes. Ikke gjenta deg selv på tvers av seksjoner."
    if words <= 9000:
        return "Hvert tema skal ha sin egen seksjon. Alle argumenter, nyanser, beslutninger og oppfølgingspunkter skal med. Ikke gjenta informasjon. Ikke avslutt for tidlig."
    return "Hvert tema skal ha sin egen seksjon. Alle argumenter, beslutninger, tall og oppfølgingspunkter skal dekkes fullstendig og presist – én gang per seksjon, ikke mer."


# ----------------------------
# Speaker extraction helper
# ----------------------------
def _extract_diarized_speakers(transcription: str) -> List[str]:
    """Plukker ut 'Person N'-etiketter fra diarisert transkripsjon."""
    found = set(re.findall(r"^(Person\s+\d+):", transcription or "", re.MULTILINE))
    return sorted(found, key=lambda x: int(re.search(r"\d+", x).group()))


def _build_deltakere_hint(transcription: str) -> str:
    """Lager en deltaker-instruksjon til modellen.
    Hvis transkripsjon har diariserte talere, brukes disse. Ellers: forbud mot å finne på navn."""
    speakers = _extract_diarized_speakers(transcription)
    if speakers:
        return (
            f"Følgende talere er identifisert i transkripsjonen: {', '.join(speakers)}. "
            "Bruk disse etikett-navnene. Ikke legg til andre navn."
        )
    return (
        "Skriv KUN navn som eksplisitt nevnes med navn i transkripsjonen. "
        "IKKE finn på, gjett eller legg til navn som ikke finnes i teksten. "
        "Hvis ingen navn fremgår: Ikke spesifisert."
    )


# ----------------------------
# Public API
# ----------------------------
def create_meeting_minutes(
    transcription: str,
    mode: str = "standard",
    meeting_date_str: str = "",
) -> str:
    """
    Lager referat basert på transkripsjon.

    mode: standard | møtereferat | mil
    """
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

    if ENABLE_TRANSCRIPTION_NORMALIZATION:
        transcription = _normalize_transcription_no_semantic_change(transcription)

    # Valider og normaliser modus-streng — fall tilbake til "standard" ved ukjent verdi
    mode_key = (mode or "standard").strip().lower() or "standard"
    if mode_key not in MODE_PROMPTS:
        mode_key = "standard"

    date_line = (meeting_date_str or "").strip() or "Ikke spesifisert."
    length_req = _length_requirements(transcription, mode_key)
    deltakere_hint = _build_deltakere_hint(transcription)

    chunks = _chunk_text_by_lines(transcription, max_chars=SUMMARY_CHUNK_MAX_CHARS)
    # Hvis transkripsjon passer i én chunk, kan vi hoppe over notes-passet og gå direkte til final
    use_direct_final = ENABLE_DIRECT_FINAL_SINGLE_CHUNK and len(chunks) == 1

    combined_notes = ""
    source_block = ""

    if use_direct_final:
        # Transkripsjon sendes uforandret som kilde til final-pass
        source_block = f"KILDETRANSKRIPSJON:\n{transcription}"
    else:
        # Fler-chunk-flyt: generer delnotater per chunk, komprimer, send til final-pass
        notes: List[str] = []
        for i, ch in enumerate(chunks, start=1):
            notes_instr = MODE_PROMPTS[mode_key]["notes"]
            prompt = f"""
Du får DEL {i}/{len(chunks)} av en transkripsjon. Transkripsjonen kan inneholde linjer som starter med "Person N:".

Modus: {mode_key}
Dato: {date_line}

Oppgave:
- {notes_instr}
- Punktvis og konkret, kun fakta fra teksten.
- Ta med alle vesentlige detaljer fra denne delen (ikke gjør det for kort).
- Skill tydelig mellom: Diskusjon, Forslag, Beslutninger, Ansvar, Risiko.
- Ikke gjett. Hvis uklart: skriv 'Ikke spesifisert'.
- Ikke utvid forkortelser eller egennavn.

Returner i denne strukturen (nøyaktig):
Temaer:
- ...

Diskusjon:
- ...

Forslag:
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
            notes.append(_postprocess_norwegian_spelling(_ollama_generate(prompt, temperature=0.2, num_predict=NOTES_NUM_PREDICT)))

        combined_notes = "\n\n".join(
            f"DELNOTATER {idx}:\n{txt}" for idx, txt in enumerate(notes, start=1)
        )

        # Komprimer delnotatene til én kompakt blokk for å redusere prompt-størrelse i final-passet
        if ENABLE_NOTES_COMPRESSION and len(chunks) > 1:
            compressed = _compress_notes_for_final(combined_notes, mode_key)
            # DEBUG — fjern når ferdig
            with open("debug_notes.txt", "w", encoding="utf-8") as _dbg:
                _dbg.write("=== KOMPRIMERTE DELNOTATER ===\n")
                _dbg.write(compressed)
                _dbg.write("\n=== SLUTT DELNOTATER ===\n")
            source_block = f"KOMPRIMERTE DELNOTATER:\n{compressed}"
        else:
            source_block = f"DELNOTATER:\n{combined_notes}"

    # ----------------------------
    # FINAL PASS per mode — velger prompt basert på modus og kaller modellen én siste gang
    # ----------------------------
    if mode_key == "mil":
        # Militært modus: todelt prompt (kartlegg saker → skriv etter militær modell)
        mil_prompt = f"""
Du skal lage et fullstendig militært referat basert på kildeteksten nedenfor.
Dato: {date_line}

STEG 1 – KARTLEGG ALLE SAKER FØRST:
Les gjennom hele kildeteksten og identifiser ALLE saker, beslutningspunkter og diskusjoner som er berørt.
Ikke hopp over noe – også korte saker, risikoer, avklaringer og åpne spørsmål skal med.

STEG 2 – SKRIV REFERATET etter militær modell:

INNLEDNING
Formål (hvorfor møtet ble holdt)
Agenda (alle saker som ble behandlet)
Rammer (føringer, ressurser, tidshorisont hvis nevnt)

HOVEDDEL – én blokk per sak fra steg 1:
Sak N – [presis tittel]
Situasjonsbeskrivelse (fakta som ble presentert)
Vurdering (risiko, konsekvens, usikkerhet)
Handlingsalternativer (hva ble vurdert)
Anbefaling (hva ble anbefalt)
Beslutning (hva ble besluttet – eller "Ikke besluttet")

AVSLUTNING
Oppsummering av beslutninger (samlet liste)
Tiltaksliste (hvem gjør hva – frist)
Risiko og oppfølging (kontrollpunkt, åpne punkter)

DEKNINGSKRAV – KRITISK:
- {length_req}
- Dekningsgrad er ALLTID viktigere enn korthet. Ikke kutt saker for å spare plass.
- Alle saker fra steg 1 skal ha sin egen blokk i hoveddelen – ingen saker skal utelates.
- Behold alle nyanser, uenigheter og perspektiver fra teksten.
- Skriv til ALT er dekket. Ikke avslutt for tidlig.
- Ikke finn på noe. Ikke bruk direkte sitater. Ikke utvid forkortelser.
- Ikke referer til at dette er basert på notater eller transkripsjon i selve teksten.

{source_block}
"""
        return _postprocess_norwegian_spelling(_ollama_generate(mil_prompt, temperature=0.2, timeout_s=OLLAMA_TIMEOUT_S_HEAVY, num_predict=FINAL_NUM_PREDICT))

    if mode_key == "standard":
        # Standard modus: fri struktur — modellen velger selv den mest naturlige oppbyggingen
        free_prompt = f"""
Du skal lage en fullstendig møtesammenfatning basert på kildeteksten nedenfor.
Dato: {date_line}
Deltakere: {deltakere_hint}

STEG 1 – KARTLEGG INNHOLDET FØRST:
Les gjennom hele kildeteksten og identifiser ALLE distinkte temaer, saker og diskusjoner som er berørt.
Ikke hopp over noe. Også korte diskusjoner, spørsmål, usikkerheter og sidekommentarer som har faglig relevans skal med.

STEG 2 – SKRIV SAMMENFATNINGEN:
Velg selv den mest naturlige strukturen basert på innholdet – ikke tving det inn i et fast skjema.
HVERT tema fra steg 1 skal ha sin egen navngitte seksjon.

Krav per seksjon:
- Overskrift som presist beskriver temaet (ikke "Diskusjon om X" – skriv hva det faktisk dreier seg om)
- Hva som ble sagt og diskutert – konkret, ikke vagt
- Hvilke argumenter og synspunkter som kom frem
- Hva som ble konkludert eller besluttet (eller at det ikke ble konkludert)
- Hvem som har ansvar og eventuelle frister hvis det fremkommer
- Hvis noe ble stemt over eller formelt avgjort: beskriv utfallet og konsekvensen, ikke bare resultatet
- Ved avstemninger med juridiske eller prosedyremessige konsekvenser: forklar ikke bare stemmetallet, men hvorfor utfallet ble som det ble og hva det betyr i praksis for saken videre
- Ikke gjenta samme informasjon under flere temaer

DEKNINGSKRAV – KRITISK:
- {length_req}
- Dekningsgrad er ALLTID viktigere enn korthet. Ikke kutt, ikke komprimer, ikke hopp over temaer for å spare plass.
- Alle temaer fra steg 1 skal ha sin egen seksjon – ingen skal slås sammen med mindre de er direkte relatert.
- Behold alle nyanser, uenigheter og perspektiver fra teksten.
- Skriv til ALT er dekket. Ikke avslutt for tidlig.
- Ikke finn på noe. Ikke bruk direkte sitater. Ikke utvid forkortelser.
- Ikke referer til at dette er basert på notater eller transkripsjon i selve teksten.

{source_block}
"""
        return _postprocess_norwegian_spelling(_ollama_generate(free_prompt, temperature=0.2, timeout_s=OLLAMA_TIMEOUT_S_HEAVY, num_predict=FINAL_NUM_PREDICT))

    # Møtereferat modus: strukturert referat med faste avslutningsseksjoner (beslutninger, oppfølging, åpne spørsmål)
    moetereferat_prompt = f"""
Du skal lage et fullstendig møtereferat basert på transkripsjonen nedenfor.
Dato: {date_line}
Deltakere: {deltakere_hint}

STEG 1 – KARTLEGG INNHOLDET FØRST:
Les gjennom hele kildeteksten og identifiser ALLE distinkte temaer, saker og diskusjoner som er berørt.
Ikke hopp over noe. Også korte diskusjoner, spørsmål, usikkerheter og sidekommentarer som har faglig relevans skal med.

STEG 2 – SKRIV REFERATET:
Skriv et strukturert møtereferat der HVERT tema fra steg 1 får sin egen navngitte seksjon.

Krav per seksjon:
- Overskrift som presist beskriver temaet (ikke bare "Diskusjon 1" eller "Tema A")
- Hva som ble sagt og diskutert – konkret, ikke vagt
- Hvilke argumenter og synspunkter som kom frem
- Hva som ble konkludert eller besluttet (eller at det ikke ble konkludert)
- Hvem som har ansvar og eventuelle frister hvis det fremkommer

AVSLUTNING – legg alltid til disse seksjonene til slutt:
Beslutninger (samlet liste over alle faktiske beslutninger fra møtet – hvis ingen: "Ingen beslutninger ble fattet")
Oppfølgingspunkter (hvem gjør hva, frist hvis nevnt – hvis ingen: "Ikke spesifisert")
Åpne spørsmål / uavklarte punkter (hvis noen fremkommer)

DEKNINGSKRAV – KRITISK:
- {length_req}
- Dekningsgrad er ALLTID viktigere enn korthet. Ikke kutt, ikke komprimer, ikke hopp over temaer for å spare plass.
- Hvert tema som er diskutert i møtet skal ha sin egen seksjon – ingen temaer skal slås sammen med mindre de er direkte relatert.
- Behold alle nyanser, uenigheter og perspektiver som fremkommer i teksten.
- Skriv til ALT er dekket. Ikke avslutt for tidlig.
- Ikke bruk "Diskutert om..." som beskrivelse – skriv hva som faktisk ble diskutert.
- Ikke finn på noe. Ikke bruk direkte sitater. Ikke utvid forkortelser.
- Ikke referer til at dette er basert på notater eller transkripsjon i selve teksten.

{source_block}
"""
    return _postprocess_norwegian_spelling(_ollama_generate(moetereferat_prompt, temperature=0.2, timeout_s=OLLAMA_TIMEOUT_S_HEAVY, num_predict=FINAL_NUM_PREDICT))
