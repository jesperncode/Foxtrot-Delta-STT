from __future__ import annotations

import os
from typing import List

import requests


# ----------------------------
# Ollama config
# ----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"

MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

# Default timeouts (seconds)
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "300"))
OLLAMA_TIMEOUT_S_HEAVY = int(os.getenv("OLLAMA_TIMEOUT_S_HEAVY", "900"))

# Chunking
SUMMARY_CHUNK_MAX_CHARS = int(os.getenv("SUMMARY_CHUNK_MAX_CHARS", "16000"))
ENABLE_DIRECT_FINAL_SINGLE_CHUNK = os.getenv("ENABLE_DIRECT_FINAL_SINGLE_CHUNK", "0").strip() == "1"

# Optional normalization (you said you likely drop this, but kept for completeness)
ENABLE_TRANSCRIPTION_NORMALIZATION = os.getenv("ENABLE_TRANSCRIPTION_NORMALIZATION", "0").strip() == "1"

# NEW: compress notes before final (big win for MIL/OKONOMI/OPPDRAG)
ENABLE_NOTES_COMPRESSION = os.getenv("ENABLE_NOTES_COMPRESSION", "1").strip() == "1"
NOTES_COMPRESSION_MAX_CHARS = int(os.getenv("NOTES_COMPRESSION_MAX_CHARS", "18000"))


SYSTEM_RULES = (
    "DU ER EN PROFESJONELL MØTEREFERENT FOR NORSKE VIRKSOMHETER.\n\n"
    "Språkkrav:\n"
    "- 100 % norsk bokmål (Norge).\n"
    "- Ikke bruk danske eller svenske ordformer.\n"
    "- Bruk offisiell norsk rettskrivning.\n"
    "- Hvis en skrivemåte kan være dansk/svensk, velg norsk bokmål.\n"
    "- Eksempel: 'Russland' (ikke 'Rusland').\n\n"
    "Forkortelser og egennavn:\n"
    "- Behold ALLE forkortelser nøyaktig slik de står (f.eks. FFI, FOH, HV, NATO, ISR).\n"
    "- Ikke utvid, forklar eller oversett forkortelser.\n"
    "- Ikke endre skrivemåten på forkortelser (store/små bokstaver, punktum, bindestrek).\n"
    "- Behold egennavn (steder, avdelinger, personer) nøyaktig slik de står.\n\n"
    "Innholdskrav:\n"
    "- Bruk kun informasjon som eksplisitt finnes i teksten.\n"
    "- Ikke tolk intensjoner som ikke er sagt.\n"
    "- Ikke finn på beslutninger eller fakta.\n"
    "- Hvis noe ikke fremgår: skriv 'Ikke spesifisert'.\n\n"
    "Kvalitetskrav:\n"
    "- Skriv profesjonelt og presist.\n"
    "- Ingen fyllord eller gjentakelser.\n"
    "- Ingen direkte sitater fra transkripsjonen, med mindre modus eksplisitt krever sitater.\n"
)


MODE_PROMPTS = {
    "standard": {
        "notes": (
            "Identifiser hovedtemaer, konkrete diskusjonspunkter, eventuelle beslutninger, "
            "ansvar og åpne spørsmål. Skill tydelig mellom hva som er diskutert og hva som er besluttet."
        ),
        "final": (
            "Lag et strukturert og profesjonelt møtereferat som gir full oversikt over "
            "hva som ble diskutert, hva som ble besluttet, og hva som krever oppfølging."
        ),
    },
    "ide": {
        "notes": (
            "Identifiser alle foreslåtte ideer, alternative løsninger, argumenter for og imot, "
            "samt usikkerhet og åpne problemstillinger. Skill klart mellom forslag og faktiske beslutninger."
        ),
        "final": (
            "Lag et strukturert idéreferat som grupperer: foreslåtte ideer, argumenter og vurderinger, "
            "fordeler og ulemper, åpne spørsmål og anbefalte videre undersøkelser eller tiltak."
        ),
    },
    "beslutning": {
        "notes": (
            "Identifiser eksplisitte beslutninger, alternativer som ble vurdert, begrunnelser, "
            "eventuelle uenigheter, ansvarlige personer og frister."
        ),
        "final": (
            "Lag et beslutningsreferat med tydelig struktur: hva som ble besluttet, hvilke alternativer som ble vurdert, "
            "begrunnelse for valgt løsning, hvem som har ansvar, frister og eventuelle uavklarte punkter."
        ),
    },
    "status": {
        "notes": (
            "Identifiser status per tema eller prosjekt: hva er fullført, hva pågår, hva er forsinket, "
            "risikoer og blokkeringer, avhengigheter og neste milepæl."
        ),
        "final": (
            "Lag et strukturert statusreferat som gir oversikt over fremdrift, risiko, avvik, ansvar "
            "og konkrete neste steg. Skal kunne brukes direkte i styringsmøte."
        ),
    },
    "god": {
        "notes": (
            "Identifiser hovedtemaer, viktigste diskusjonspunkter, eventuelle beslutninger, "
            "ansvar og neste steg. Prioriter det viktigste, men behold nok detaljer til at oppsummeringen blir nyttig."
        ),
        "final": (
            "Lag en god og konsis oppsummering av møtet. Den skal ikke være kjempelang, "
            "men den må dekke de viktigste temaene, beslutningene og oppfølgingen."
        ),
    },
    "fri": {
        "notes": (
            "Identifiser hovedtemaer, nøkkelpunkter, mulige beslutninger, ansvar, risiko og åpne spørsmål. "
            "Trekk ut en naturlig struktur fra innholdet selv om samtalen er ustrukturert."
        ),
        "final": (
            "Lag en strukturert møtesammenfatning der du selv velger den mest hensiktsmessige strukturen "
            "basert på innholdet, ikke et fast skjema."
        ),
    },
    "ir": {
        "notes": (
            "Identifiser hovedtemaer i intervjuet. Fjern småprat og irrelevante digresjoner. "
            "Slå sammen gjentakelser og presiseringer. Dersom intervjuobjektet endrer meningsinnhold "
            "i et tema underveis, behold den siste og gjeldende versjonen. "
            "Behold alle navn fullt ut slik de fremkommer i teksten. "
            "Marker eventuelle direkte sitater tydelig (ordrett)."
        ),
        "final": (
            "Lag et tematisk intervjureferat (IR) etter fastsatte krav for språk, struktur og avsnittsform."
        ),
    },
    "mil": {
        "notes": (
            "Trekk ut hovedsaker og beslutningspunkter i en strukturert, disiplinert form. "
            "For hver sak: fakta/situasjon, vurdering/risiko, handlingsalternativer, anbefaling, beslutning og tiltak."
        ),
        "final": (
            "Lag et strukturert referat etter militær modell (formål/agenda/rammer, sak-for-sak med fakta–vurdering–alternativer–anbefaling–beslutning, og avslutning med tiltaksliste)."
        ),
    },
    "oppdrag": {
        "notes": (
            "Trekk ut intensjon/ønsket effekt, rammer, uavklarte premisser, muligheter/risiko, "
            "2–3 handlingsalternativer, anbefalt retning, og videre oppdrag (hvem utreder hva, tidslinje)."
        ),
        "final": (
            "Lag et referat etter intensjonsbasert/oppdragstenkning: intensjon, rammer, felles situasjonsforståelse, "
            "drøfting, alternativutvikling, anbefalt retning, videre oppdrag og tidslinje."
        ),
    },
    "fremdrift": {
        "notes": (
            "Trekk ut status siden sist (leveranser/avvik), 1–3 kritiske saker, hindringer, "
            "konkrete tiltak, prioritering, ansvar/forpliktelser, risiko fremover og kontrollpunkt."
        ),
        "final": (
            "Lag et fremdriftsorientert referat med fokus på leveranser, avvik, hindringer, tiltak, prioritering, ansvar og neste kontrollpunkt."
        ),
    },
    "okonomi": {
        "notes": (
            "Trekk ut formål, styringskontekst, datagrunnlag, forbruk vs budsjett, avvik (kr/%), "
            "prognose, årsaker, risiko, tiltak/omprioriteringer, beslutning, rapportering og frister."
        ),
        "final": (
            "Lag et økonomi- og styringsreferat: status/avvik/prognose, årsaker, risiko, tiltak, beslutning, rapportering og kontrollpunkt."
        ),
    },
}


# ----------------------------
# Core helpers
# ----------------------------
def _ollama_generate(prompt: str, temperature: float = 0.2, timeout_s: int | None = None) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": float(temperature)},
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
    return (data.get("response") or "").strip()


def _chunk_text_by_lines(text: str, max_chars: int) -> List[str]:
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
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

    prompt = f"""{SYSTEM_RULES}

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

    prompt = f"""{SYSTEM_RULES}

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
    compressed = _ollama_generate(prompt, temperature=0.0, timeout_s=OLLAMA_TIMEOUT_S_HEAVY)
    return _postprocess_norwegian_spelling(compressed)


def _length_requirements(transcription: str, mode: str) -> str:
    words = len((transcription or "").split())

    if mode == "ir":
        if words <= 700:
            return "Sikt på omtrent 280–500 ord totalt."
        if words <= 1800:
            return "Sikt på omtrent 500–800 ord totalt."
        return "Sikt på omtrent 800–1200 ord totalt."

    if words <= 500:
        return "Sikt på omtrent 260–430 ord totalt."
    if words <= 1200:
        return "Sikt på omtrent 430–720 ord totalt."
    if words <= 2500:
        return "Sikt på omtrent 720–1150 ord totalt."
    return "Sikt på omtrent 1150–1700 ord totalt."


# ----------------------------
# Public API
# ----------------------------
def create_meeting_minutes(
    transcription: str,
    mode: str = "standard",
    meeting_date_str: str = "",
    role: str = "Intervjuobjekt",
) -> str:
    """
    Lager referat basert på transkripsjon.

    mode:
      - standard | ide | beslutning | status | god | fri | ir | mil | oppdrag | fremdrift | okonomi
    role:
      - Brukes kun av IR-modus. Typiske verdier: "Vitne", "Varsler", "Omvarslede".
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

    mode_key = (mode or "standard").strip().lower() or "standard"
    if mode_key not in MODE_PROMPTS:
        mode_key = "standard"

    date_line = (meeting_date_str or "").strip() or "Ikke spesifisert."
    role_line = (role or "Intervjuobjekt").strip() or "Intervjuobjekt"
    length_req = _length_requirements(transcription, mode_key)

    chunks = _chunk_text_by_lines(transcription, max_chars=SUMMARY_CHUNK_MAX_CHARS)
    use_direct_final = ENABLE_DIRECT_FINAL_SINGLE_CHUNK and len(chunks) == 1

    combined_notes = ""
    source_block = ""

    if use_direct_final:
        source_block = f"KILDETRANSKRIPSJON:\n{transcription}"
    else:
        notes: List[str] = []
        for i, ch in enumerate(chunks, start=1):
            notes_instr = MODE_PROMPTS[mode_key]["notes"]
            prompt = f"""{SYSTEM_RULES}

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
            notes.append(_postprocess_norwegian_spelling(_ollama_generate(prompt, temperature=0.2)))

        combined_notes = "\n\n".join(
            f"DELNOTATER {idx}:\n{txt}" for idx, txt in enumerate(notes, start=1)
        )

        # NEW: compress notes for heavy modes (and optionally other structured modes)
        heavy_modes_for_compression = {"mil", "oppdrag", "okonomi", "fremdrift", "status"}
        if ENABLE_NOTES_COMPRESSION and mode_key in heavy_modes_for_compression:
            compressed = _compress_notes_for_final(combined_notes, mode_key)
            source_block = f"KOMPRIMERTE DELNOTATER:\n{compressed}"
        else:
            source_block = f"DELNOTATER:\n{combined_notes}"

    # ----------------------------
    # FINAL PASS per mode
    # ----------------------------
    if mode_key == "ir":
        ir_prompt = f"""{SYSTEM_RULES}

Dette er et intervjureferat (IR).

Rolle på intervjuobjekt:
- {role_line}

Hovedbestilling:
Referatet skal være en tematisk oppsummering av innholdet i det transkriberte intervjuet.

Generelle krav:
- Svar kun på norsk bokmål (Norge).
- Referatet skal gjengi meningsinnhold, men ikke nødvendigvis ordlyden.
- Referatet skal være kort og konsist, men uten at vesentlige detaljer går tapt.
- {length_req}
- Irrelevante digresjoner, småprat og fyllord skal fjernes.
- Unngå gjentakelser.
- Temaer som gjentas, utdypes eller presiseres, skal slås sammen.
- Dersom intervjuobjektet bruker mange ord eller ikke svarer direkte, oppsummer slik at meningsinnholdet kommer tydelig frem.
- Dersom intervjuobjektet endrer meningsinnhold i et tema underveis, gjengi den siste og gjeldende versjonen.
- Referatet skal skrives i preteritum, med unntak av direkte sitater eller der presens/futurum må brukes for å bevare meningen.
- Direkte sitater: *«sitat»*.

Avsnittsform:
- Hvert avsnitt skal begynne med én av disse:
  • «{role_line} forklarte …»
  • «På spørsmål om [gjengivelse av spørsmålet] forklarte {role_line} …»
- Ikke bruk andre startformer.

Navn:
- Alle navn som nevnes må skrives fullt ut, og i den rolle/kontekst/sammenheng de fremkommer.

Struktur:
1) Innledning
   - Hvem som var til stede (hvis fremkommer).
   - Bakgrunnen for intervjuet (hvis fremkommer).
   - Intervjuobjektets bakgrunn (hvis fremkommer).
2) Hoveddel
   - Organiser etter hovedtemaer.
   - Hvert tema skal ha en kort overskrift som presist gjenspeiler temaets innhold.
3) Avslutning
   - Om intervjuobjektet hadde spørsmål eller ønsker (hvis fremkommer).
   - Intervjuerens forklaring på hva som skjer videre (hvis fremkommer).

Forbud:
- Ikke vurder troverdighet.
- Ikke tolk motiv eller intensjon.
- Ikke legg til fakta.
- Hvis informasjon mangler: skriv 'Ikke spesifisert' der det er relevant.

{source_block}
"""
        return _postprocess_norwegian_spelling(_ollama_generate(ir_prompt, temperature=0.2, timeout_s=OLLAMA_TIMEOUT_S_HEAVY))

    if mode_key == "mil":
        mil_prompt = f"""{SYSTEM_RULES}

Modus: {mode_key}
Dato: {date_line}

Møtestruktur: Strukturert og disiplinert (militær modell)

Krav:
- Svar kun på norsk bokmål (Norge)
- Ikke utvid eller forklar forkortelser
- Ikke bruk direkte sitater
- Ikke finn på noe
- {length_req}
- Prioriter dekningsgrad over korthet der det trengs
- Ikke referer til delnotater i svaret

FORMAT (bruk disse overskriftene):

INNLEDNING
Formål
Agenda
Rammer

HOVEDDEL
Sak 1 – [kort tittel]
Situasjonsbeskrivelse (fakta)
Vurdering (risiko/konsekvens)
Handlingsalternativer
Anbefaling
Beslutning

Sak 2 – [kort tittel]
... (gjenta ved behov)

AVSLUTNING
Oppsummering av beslutninger
Tiltaksliste (hvem gjør hva – frist)
Risiko og oppfølging (kontrollpunkt)

{source_block}
"""
        return _postprocess_norwegian_spelling(_ollama_generate(mil_prompt, temperature=0.2, timeout_s=OLLAMA_TIMEOUT_S_HEAVY))

    if mode_key == "oppdrag":
        oppdrag_prompt = f"""{SYSTEM_RULES}

Modus: {mode_key}
Dato: {date_line}

Møtestruktur: Intensjonsbasert (oppdragstenkning)

Krav:
- Svar kun på norsk bokmål (Norge)
- Ikke utvid eller forklar forkortelser
- Ikke bruk direkte sitater
- Ikke finn på noe
- {length_req}
- Ikke referer til delnotater i svaret

FORMAT (bruk disse overskriftene):

INNLEDNING
Intensjon (ønsket effekt)
Rammer (fast vs handlingsrom)
Forventninger

HOVEDDEL
Felles situasjonsforståelse (fakta/premisser)
Drøfting (muligheter/risiko/konsekvens)
Alternativutvikling (2–3 realistiske alternativer)
Anbefaling og retning (enighet/beslutning)

AVSLUTNING
Felles forståelse (hva er vi enige om / rest-usikkerhet)
Videre oppdrag (hvem utreder hva / hvem beslutter videre)
Tidslinje (neste milepæl)

{source_block}
"""
        return _postprocess_norwegian_spelling(_ollama_generate(oppdrag_prompt, temperature=0.2, timeout_s=OLLAMA_TIMEOUT_S_HEAVY))

    if mode_key == "fremdrift":
        fremdrift_prompt = f"""{SYSTEM_RULES}

Modus: {mode_key}
Dato: {date_line}

Møtestruktur: Effektivitets- og fremdriftsorientert

Krav:
- Svar kun på norsk bokmål (Norge)
- Ikke utvid eller forklar forkortelser
- Ikke bruk direkte sitater
- Ikke finn på noe
- {length_req}
- Ikke referer til delnotater i svaret

FORMAT (bruk disse overskriftene):

INNLEDNING
Status siden sist (leveranser/avvik)
Hovedfokus i dag (1–3 kritiske saker)
Suksesskriterium for møtet

HOVEDDEL
Fremdriftsanalyse (hva hindrer progresjon)
Tiltaksdiskusjon (konkret og gjennomførbart)
Prioritering (først / kan vente)
Forpliktelse (bekreftet ansvar)

AVSLUTNING
Tiltaksliste (ansvar, frist, avhengigheter)
Risikovurdering fremover
Kontrollpunkt (hvordan/når følges det opp)

{source_block}
"""
        return _postprocess_norwegian_spelling(_ollama_generate(fremdrift_prompt, temperature=0.2, timeout_s=OLLAMA_TIMEOUT_S_HEAVY))

    if mode_key == "okonomi":
        okonomi_prompt = f"""{SYSTEM_RULES}

Modus: {mode_key}
Dato: {date_line}

Møtestruktur: Økonomi og styring

Krav:
- Svar kun på norsk bokmål (Norge)
- Ikke utvid eller forklar forkortelser
- Ikke bruk direkte sitater
- Ikke finn på noe
- {length_req}
- Ikke referer til delnotater i svaret

FORMAT (bruk disse overskriftene):

INNLEDNING
Formål
Styringskontekst
Rapporteringsgrunnlag (datagrunnlag/forutsetninger)

HOVEDDEL
Økonomisk status (forbruk vs budsjett)
Avvik (kr og %)
Prognose ved årsslutt
Årsaksanalyse
Risikoanalyse (over-/mindreforbruk, konsekvens)
Tiltak (kutt/omprioritering/styrking)
Beslutning
Rapportering videre (styringslinje)

AVSLUTNING
Tiltaksliste (ansvarlig – frist)
Frister (oppdatert prognose/rapportering)
Kontrollpunkt

{source_block}
"""
        return _postprocess_norwegian_spelling(_ollama_generate(okonomi_prompt, temperature=0.2, timeout_s=OLLAMA_TIMEOUT_S_HEAVY))

    if mode_key == "fri":
        free_final_instr = MODE_PROMPTS[mode_key]["final"]
        free_prompt = f"""{SYSTEM_RULES}

Modus: {mode_key}
Dato: {date_line}

{free_final_instr}

Mål:
- Lag en tydelig og naturlig struktur basert på innholdet.
- Når møtet mangler klar agenda, organiser i logiske temaer som gjør teksten lett å bruke i ettertid.
- Prioriter dekningsgrad over korthet der det trengs.
- {length_req}

Krav:
- Svar kun på norsk bokmål (Norge)
- Ikke utvid eller forklar forkortelser
- Ikke bruk direkte sitater
- Ikke finn på noe
- Hvis informasjon mangler: Ikke spesifisert
- Ikke referer til delnotater i svaret

{source_block}
"""
        return _postprocess_norwegian_spelling(_ollama_generate(free_prompt, temperature=0.2, timeout_s=OLLAMA_TIMEOUT_S_HEAVY))

    # STANDARD / IDE / BESLUTNING / STATUS / GOD  (valg C: mer dekningsgrad)
    final_instr = MODE_PROMPTS[mode_key]["final"]
    final_prompt = f"""{SYSTEM_RULES}

Modus: {mode_key}
Dato: {date_line}

{final_instr}

KRITISK:
- Prioriter dekningsgrad over korthet når teksten er lang eller faglig.
- Ikke komprimer bort forklaringer som trengs for å forstå temaet.
- Hvis et tema har nyanser/perspektiver: behold dem.
- Ikke tving frem beslutninger/oppgaver hvis de ikke finnes.
- Alt må være forankret i teksten.
- {length_req}
- Hvis informasjon mangler: 'Ikke spesifisert'.
- Ikke referer til delnotater i svaret.

FORMAT (bruk nøyaktig disse overskriftene):

Tittel
- Kort og beskrivende

Dato
- {date_line}

Deltakere
- Navn hvis nevnt
- Hvis ikke: Ikke spesifisert

Agenda / Tema
- Strukturert punktliste

Oppsummering av diskusjon
- Strukturert etter tema
- Skill mellom drøfting og konklusjon der det faktisk fremkommer

Beslutninger
- Punktvis, kun faktiske beslutninger
- Hvis ingen: Ingen beslutninger ble fattet

Oppfølgingspunkter
- Punktvis
- Inkluder ansvarlig og frist hvis nevnt
- Hvis ingen: Ikke spesifisert

Neste steg
- Konkret og handlingsrettet
- Hvis ikke fremkommer: Ikke spesifisert

Risikoer / Avklaringer
- Punktvis
- Hvis ikke fremkommer: Ikke spesifisert

{source_block}
"""
    # Standard-modus bør vanligvis ikke trenge heavy timeout, men med lange møter kan det fortsatt være nyttig.
    # Vi bruker "heavy" for alle final-pass for å redusere 500/timeout.
    return _postprocess_norwegian_spelling(_ollama_generate(final_prompt, temperature=0.2, timeout_s=OLLAMA_TIMEOUT_S_HEAVY))
