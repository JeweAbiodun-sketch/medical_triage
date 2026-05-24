"""
medical_agent_server.py
FastAPI server wrapping the full Medical Triage LangGraph pipeline.
n8n calls this via HTTP POST /triage

Run with:
    python medical_agent_server.py

Test with:
    curl -X POST http://localhost:8001/triage \
      -H "Content-Type: application/json" \
      -d '{"message": "I have fever and headache for 3 days", "channel": "telegram", "chat_id": "8138298582"}'
"""

import os, json, time, uuid, re, sys, requests, difflib
from datetime import datetime, timezone
from typing import Optional, List, Dict, TypedDict
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
from pinecone import Pinecone
from pinecone.control import ServerlessSpec
import cohere
from langgraph.graph import StateGraph, END

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST", "").strip()
COHERE_API_KEY     = os.getenv("COHERE_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

assert OPENAI_API_KEY,   "OPENAI_API_KEY not set"
assert PINECONE_API_KEY, "PINECONE_API_KEY not set"
assert COHERE_API_KEY,   "COHERE_API_KEY not set"

openai_client  = OpenAI(api_key=OPENAI_API_KEY)
pc             = Pinecone(api_key=PINECONE_API_KEY)
cohere_client  = cohere.Client(COHERE_API_KEY)

PINECONE_INDEX_NAME = "medical-triage-agent"
EMBED_MODEL         = "text-embedding-3-small"
RESEARCH_MODEL      = "gpt-4o-mini"
REPORT_MODEL        = "gpt-4o"
PUBMED_BASE_URL     = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SMS_API_URL         = os.getenv("SMS_API_URL", "")
SMS_API_KEY         = os.getenv("SMS_API_KEY", "")
SMS_SENDER_ID       = os.getenv("SMS_SENDER_ID", "TriageAI")

LANGUAGE_KEYWORDS = {
    "hausa": [
        "ina", "zafi", "ciwo", "matsala", "dari", "numfashi", "jini", "jiki"
    ],
    "igbo": [
        "ahụ", "ọrịa", "mgbu", "isi", "ọbara", "ụra", "ike", "afọ"
    ],
    "yoruba": [
        "mo ni", "irora", "ara", "ori", "eje", "aisan", "isun", "ìmú"
    ],
}

SUPPORTED_LANGUAGES = ("english", "hausa", "igbo", "yoruba")

NIGERIAN_COMMON_CONDITIONS = {
    "malaria": {
        "clues": ["fever", "chills", "headache", "body ache", "mosquito"],
        "facility": "clinic",
        "notes": "Common in many parts of Nigeria; consider malaria testing when fever is present.",
    },
    "typhoid": {
        "clues": ["fever", "abdominal pain", "diarrhea", "weakness"],
        "facility": "clinic",
        "notes": "Common differential for prolonged fever and abdominal symptoms.",
    },
    "hypertension": {
        "clues": ["headache", "blurred vision", "chest pain", "bp", "blood pressure"],
        "facility": "clinic",
        "notes": "Hypertension is a frequent chronic condition and can present silently.",
    },
    "diabetes": {
        "clues": ["thirst", "urinating", "weight loss", "blurred vision"],
        "facility": "clinic",
        "notes": "Check glucose when polyuria, polydipsia, or unexplained weight loss is reported.",
    },
    "gastroenteritis": {
        "clues": ["vomiting", "diarrhea", "stomach", "abdominal", "dehydration"],
        "facility": "clinic",
        "notes": "Hydration status matters; escalate if unable to keep fluids down.",
    },
    "asthma": {
        "clues": ["wheezing", "shortness of breath", "chest tightness", "cough"],
        "facility": "clinic",
        "notes": "Respiratory distress needs urgent assessment if severe or worsening.",
    },
}

FACILITY_DIRECTORY = [
    {
        "name": "General Hospital, Lagos",
        "city": "lagos",
        "state": "lagos",
        "level": "secondary",
        "type": "hospital",
        "services": ["emergency", "general medicine", "paediatrics"],
        "contact": "Nearest public secondary facility",
    },
    {
        "name": "University College Hospital",
        "city": "ibadan",
        "state": "oyo",
        "level": "tertiary",
        "type": "hospital",
        "services": ["specialist care", "emergency", "internal medicine"],
        "contact": "Tertiary referral center",
    },
    {
        "name": "National Hospital Abuja",
        "city": "abuja",
        "state": "fct",
        "level": "tertiary",
        "type": "hospital",
        "services": ["emergency", "specialist care"],
        "contact": "Federal tertiary referral center",
    },
    {
        "name": "Aminu Kano Teaching Hospital",
        "city": "kano",
        "state": "kano",
        "level": "tertiary",
        "type": "hospital",
        "services": ["emergency", "specialist care"],
        "contact": "Teaching hospital",
    },
    {
        "name": "University of Nigeria Teaching Hospital",
        "city": "enugu",
        "state": "enugu",
        "level": "tertiary",
        "type": "hospital",
        "services": ["emergency", "specialist care"],
        "contact": "Teaching hospital",
    },
    {
        "name": "Nearest Primary Health Centre",
        "city": "any",
        "state": "any",
        "level": "primary",
        "type": "clinic",
        "services": ["basic consultation", "vaccination", "health education"],
        "contact": "Use local PHC directory or ward health office",
    },
]

def setup_pinecone_index(index_name: str, dims: int = 1536, max_attempts: int = 5):
    def _normalize_host(host: Optional[str]) -> Optional[str]:
        host = (host or "").strip()
        if not host:
            return None
        # Pinecone needs the full data-plane host, not the index name.
        if "pinecone.io" not in host:
            return None
        return host

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            existing = [idx.name for idx in pc.list_indexes()]
            if index_name not in existing:
                print(f"[PINECONE] Creating index {index_name!r} ({dims} dims)...")
                pc.create_index(
                    name=index_name,
                    dimension=dims,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

            while True:
                desc = pc.describe_index(index_name)
                if desc.status["ready"]:
                    host = _normalize_host(getattr(desc, "host", None))
                    if not host and isinstance(desc, dict):
                        host = _normalize_host(desc.get("host"))
                    if not host:
                        host = _normalize_host(PINECONE_INDEX_HOST)
                    if not host:
                        raise RuntimeError(
                            f"Pinecone index {index_name!r} is ready, but no host was returned. "
                            "Set PINECONE_INDEX_HOST to the full Pinecone data-plane host "
                            "(for example, a host ending in .pinecone.io)."
                        )
                    print(f"[PINECONE] Index ready: {index_name} ({host})")
                    return pc.Index(host=host)
                print("[PINECONE] Waiting for index to become ready...")
                time.sleep(2)
        except Exception as e:
            last_error = e
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Failed to initialize Pinecone index {index_name!r} after {max_attempts} attempts. "
                    "This usually means a transient network/SSL problem or a blocked outbound connection."
                ) from e
            delay = min(2 ** attempt, 15)
            print(f"[PINECONE] Setup attempt {attempt}/{max_attempts} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError(f"Failed to initialize Pinecone index {index_name!r}: {last_error}")


# Setup Pinecone
pinecone_index = setup_pinecone_index(PINECONE_INDEX_NAME, 1536)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════
class MedicalAgentState(TypedDict):
    session_id:        str
    ticket_id:         str
    initiated_at:      str
    channel:           str
    chat_id:           str
    phone_number:      Optional[str]
    user_mode:         str
    preferred_language: Optional[str]
    detected_language: Optional[str]
    facility_location: Optional[str]
    facility_recommendation: Optional[Dict[str, str]]
    offline_mode:      bool
    raw_message:       str
    corrected_message: Optional[str]
    symptoms:          List[str]
    duration:          Optional[str]
    age:               Optional[str]
    medications:       List[str]
    status:            str
    errors:            List[str]
    raw_research:      Optional[str]
    research_chunks:   Optional[List[str]]
    pinecone_ids:      Optional[List[str]]
    clinical_namespace: Optional[str]
    retrieved_chunks:  Optional[List[str]]
    reranked_chunks:   Optional[List[str]]
    urgency_level:     Optional[str]
    urgency_score:     Optional[int]
    differential:      Optional[List[str]]
    red_flags:         Optional[List[str]]
    retry_count:       int
    triage_report:     Optional[Dict[str, str]]
    nutrition_advice:  Optional[str]
    home_remedies:     Optional[str]
    follow_up_sent:    bool
    report_ready:      bool
    workflow_path:     List[str]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
CLINICIAN_KEYWORDS = [
    "i am a doctor", "i am a nurse", "i am a clinician", "i am a pharmacist",
    "my patient", "the patient", "presenting with", "chief complaint",
    "differential", "treatment protocol", "drug of choice", "nhis provider"
]
RED_FLAG_SYMPTOMS = {
    "cardiac":      ["chest pain", "cannot breathe", "shortness of breath"],
    "neurological": ["seizure", "convulsion", "fits", "unconscious", "stroke"],
    "haemorrhage":  ["vomiting blood", "blood in stool", "bleeding heavily"],
    "sepsis":       ["temperature 40", "not responding", "shaking badly"],
}
NIGERIAN_NUTRITION = {
    "malaria":      {"eat": ["Bitter leaf soup", "Citrus fruits", "Plenty water", "Light pap"],
                     "avoid_during": ["Heavy oily food", "Alcohol", "Spicy food until you feel better"],
                     "tip": "Drink 2-3 litres of water daily."},
    "typhoid":      {"eat": ["Light pap", "Oats", "Boiled yam", "Plenty water"],
                     "avoid_during": ["Pepper", "Fried food", "Street food", "Raw vegetables"],
                     "tip": "Eat small portions. Wash hands before eating."},
    "anaemia":      {"eat": ["Ugu (pumpkin leaves)", "Liver", "Beans", "Watermelon"],
                     "avoid_during": ["Tea with meals"],
                     "avoid_long_term": ["Too much alcohol"],
                     "tip": "Eat vitamin C foods with iron-rich foods."},
    "hypertension": {"eat": ["Garden egg", "Fish (grilled)", "Oats", "Banana"],
                     "avoid_long_term": ["Salt", "Seasoning cubes", "Fried foods", "Red meat", "Sugary drinks", "Alcohol"],
                     "tip": "Cook without Maggi."},
    "diabetes":     {"eat": ["Tiger nuts", "Unripe plantain", "Garden egg", "Vegetables"],
                     "avoid_long_term": ["Eba/fufu large portions", "Sugary drinks", "White bread", "Puff puff", "Sweet chin chin"],
                     "tip": "Eat at regular times."},
    "default":      {"eat": ["Plenty water", "Fruits and vegetables", "Light foods"],
                     "avoid_during": ["Alcohol", "Very oily food"],
                     "tip": "Rest and stay hydrated."},
}
RESEARCH_TOPICS = [
    "clinical presentation symptoms and diagnosis criteria",
    "WHO Nigeria treatment guidelines and protocols",
    "NHIS Nigeria coverage and treatment pathways",
    "differential diagnosis similar conditions to rule out",
    "recommended diagnostic tests and investigations",
    "drug treatment options dosages and contraindications",
    "self-care management and home treatment guidelines",
    "nutrition and dietary recommendations during illness",
]
URGENCY_ACTIONS = {
    "emergency": "Go to A&E IMMEDIATELY. Call 112 now.",
    "urgent":    "See a doctor within 24 hours.",
    "standard":  "Book a clinic appointment this week.",
    "self-care": "You can manage this at home with the guidance below.",
}

SPELLING_CORRECTIONS = {
    "feaver": "fever",
    "fiever": "fever",
    "hedache": "headache",
    "headake": "headache",
    "maleria": "malaria",
    "malaeria": "malaria",
    "tyfoid": "typhoid",
    "typoid": "typhoid",
    "diarhoea": "diarrhea",
    "diareah": "diarrhea",
    "nusea": "nausea",
    "vomitting": "vomiting",
    "dizzyness": "dizziness",
    "breething": "breathing",
    "shorness": "shortness",
    "abdomnial": "abdominal",
    "stomack": "stomach",
}

SPELLING_PHRASES = {
    r"\bshort of breath\b": "shortness of breath",
    r"\bchest pains\b": "chest pain",
    r"\bbody ache\b": "body ache",
}

SPELLING_VOCAB = sorted({
    "fever", "headache", "malaria", "typhoid", "diarrhea", "nausea", "vomiting",
    "dizziness", "breathing", "shortness", "stomach", "abdominal", "chills",
    "weakness", "fatigue", "cough", "wheezing", "pain", "rash", "sweating",
    "body", "ache", "thirst", "urinating", "weight", "loss", "blurred",
    "vision", "chest", "convulsion", "seizure", "unconscious", "stroke",
})


def detect_user_mode(msg: str) -> str:
    m = msg.lower()
    return "clinician" if any(k in m for k in CLINICIAN_KEYWORDS) else "patient"

def detect_language(msg: str, preferred: Optional[str] = None) -> str:
    if preferred:
        preferred = preferred.strip().lower()
        if preferred in SUPPORTED_LANGUAGES:
            return preferred
    lowered = msg.lower()
    for language, keywords in LANGUAGE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return language
    return "english"

def normalize_message_text(msg: str) -> str:
    text = msg
    for pattern, replacement in SPELLING_PHRASES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    def replace_word(match: re.Match) -> str:
        word = match.group(0)
        lowered = word.lower()
        if lowered in SPELLING_CORRECTIONS:
            replacement = SPELLING_CORRECTIONS[lowered]
        elif len(lowered) < 5 or not lowered.isalpha():
            return word
        elif lowered in SPELLING_VOCAB:
            return word
        else:
            close = difflib.get_close_matches(lowered, SPELLING_VOCAB, n=1, cutoff=0.9)
            if not close:
                return word
            replacement = close[0]

        if word.isupper():
            return replacement.upper()
        if word[:1].isupper():
            return replacement.capitalize()
        return replacement

    return re.sub(r"\b[A-Za-z]{4,}\b", replace_word, text)

def llm_correct_message_text(msg: str) -> str:
    prompt = (
        "Correct spelling and obvious typos in this medical triage message.\n"
        "Rules:\n"
        "- Preserve the meaning exactly.\n"
        "- Do not add new symptoms, diagnoses, or advice.\n"
        "- Keep medication names, doses, numbers, and names unchanged.\n"
        "- Return only the corrected message text.\n\n"
        f"Message: {msg}"
    )
    try:
        resp = openai_client.chat.completions.create(
            model=RESEARCH_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a careful spelling corrector for medical text."},
                {"role": "user", "content": prompt},
            ],
        )
        corrected = (resp.choices[0].message.content or "").strip()
        return corrected or msg
    except Exception as e:
        print(f"[PARSE] LLM spelling correction failed: {e}")
        return msg

def correct_message_for_parsing(msg: str) -> str:
    return normalize_message_text(llm_correct_message_text(msg))

def extract_symptoms(msg: str) -> dict:
    dur  = re.search(r"(\d+\s*(?:day|days|week|weeks|hour|hours))", msg, re.IGNORECASE)
    age  = re.search(r"(\d+)\s*(?:year|yr|years old)", msg, re.IGNORECASE)
    syms = [s.strip() for s in re.split(r"[,;]|\band\b", msg) if len(s.strip()) > 3]
    return {"symptoms": syms[:10], "duration": dur.group(1) if dur else None,
            "age": age.group(1) + " years" if age else None, "medications": []}

def detect_red_flags(msg: str) -> List[str]:
    m = msg.lower()
    return [f"{cat}: {kw}" for cat, kws in RED_FLAG_SYMPTOMS.items() for kw in kws if kw in m][:3]

def get_nutrition_advice(symptoms: List[str], differential: List[str]) -> str:
    all_text = " ".join(symptoms + differential).lower()
    for cond, data in NIGERIAN_NUTRITION.items():
        if cond != "default" and cond in all_text:
            parts = ["FOODS TO EAT:\n" + "\n".join([f"  - {f}" for f in data["eat"]])]
            if data.get("avoid_during"):
                parts.append("FOODS TO AVOID DURING ILLNESS:\n" + "\n".join([f"  - {f}" for f in data["avoid_during"]]))
            if data.get("avoid_long_term"):
                parts.append("FOODS TO AVOID LONG-TERM:\n" + "\n".join([f"  - {f}" for f in data["avoid_long_term"]]))
            parts.append(f"TIP: {data['tip']}")
            return "\n\n".join(parts)
    d = NIGERIAN_NUTRITION["default"]
    parts = ["GUIDANCE:\n" + "\n".join([f"  - {f}" for f in d["eat"]])]
    if d.get("avoid_during"):
        parts.append("FOODS TO AVOID DURING ILLNESS:\n" + "\n".join([f"  - {f}" for f in d["avoid_during"]]))
    if d.get("avoid_long_term"):
        parts.append("FOODS TO AVOID LONG-TERM:\n" + "\n".join([f"  - {f}" for f in d["avoid_long_term"]]))
    parts.append(f"TIP: {d['tip']}")
    return "\n\n".join(parts)

def match_common_condition(symptoms: List[str], differential: List[str]) -> Optional[Dict[str, str]]:
    text = " ".join(symptoms + differential).lower()
    for condition, meta in NIGERIAN_COMMON_CONDITIONS.items():
        if any(clue in text for clue in meta["clues"]):
            return {
                "condition": condition,
                "facility_type": meta["facility"],
                "notes": meta["notes"],
            }
    return None

def infer_facility(location: Optional[str], urgency: Optional[str], condition_hint: Optional[Dict[str, str]]) -> Dict[str, str]:
    loc = (location or "").strip().lower()
    if urgency == "emergency":
        return {
            "name": "Nearest Emergency Department",
            "level": "emergency",
            "type": "hospital",
            "reason": "Red flag symptoms require immediate hospital evaluation.",
        }

    if condition_hint:
        facility_type = condition_hint.get("facility_type", "clinic")
        if facility_type == "hospital":
            return {
                "name": "General Hospital or Teaching Hospital",
                "level": "secondary",
                "type": "hospital",
                "reason": condition_hint.get("notes", "Hospital-level review recommended."),
            }

    candidates = []
    if loc:
        for facility in FACILITY_DIRECTORY:
            if loc in {facility["city"], facility["state"]} or loc in facility["name"].lower():
                candidates.append(facility)
    if not candidates:
        candidates = FACILITY_DIRECTORY

    selected = candidates[0]
    return {
        "name": selected["name"],
        "level": selected["level"],
        "type": selected["type"],
        "reason": selected["contact"],
    }

def classify_urgency_rules(msg: str, red_flags: List[str], symptoms: List[str]) -> Dict[str, object]:
    lowered = msg.lower()
    score = 3
    if red_flags:
        return {"urgency_level": "emergency", "urgency_score": 10}

    if any(term in lowered for term in ["shortness of breath", "cannot breathe", "chest pain", "stroke", "unconscious"]):
        score = 9
    elif any(term in lowered for term in ["severe", "heavy bleeding", "vomiting blood", "blood in stool"]):
        score = 9
    elif any(term in lowered for term in ["fever", "high fever", "persistent fever", "worsening", "cannot eat"]):
        score = max(score, 6)
    elif any(term in lowered for term in ["pain", "rash", "headache", "cough", "cold"]):
        score = max(score, 4)

    if len(symptoms) >= 5:
        score = min(8, score + 1)

    if score >= 9:
        level = "emergency"
    elif score >= 6:
        level = "urgent"
    elif score >= 3:
        level = "standard"
    else:
        level = "self-care"
    return {"urgency_level": level, "urgency_score": score}

def search_pubmed(query: str, max_results: int = 3) -> List[Dict]:
    try:
        r = requests.get(f"{PUBMED_BASE_URL}/esearch.fcgi",
            params={"db": "pubmed", "term": f"{query}[Title/Abstract]",
                    "retmax": max_results, "retmode": "json"}, timeout=8)
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids: return []
        s = requests.get(f"{PUBMED_BASE_URL}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=8)
        result = s.json().get("result", {})
        return [{"pmid": pid, "title": result.get(pid, {}).get("title", ""),
                 "year": result.get(pid, {}).get("pubdate", "")[:4],
                 "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"} for pid in ids if pid in result]
    except Exception as e:
        print(f"[PUBMED] Error: {e}")
        return []

def send_telegram(chat_id: str, text: str):
    if not TELEGRAM_BOT_TOKEN:
        print("[TELEGRAM] Skipped: TELEGRAM_BOT_TOKEN not configured")
        return False
    try:
        import urllib.request
        for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
            payload = json.dumps({"chat_id": chat_id, "text": chunk}).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data=payload, headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req)
            time.sleep(0.3)
        return True
    except Exception as e:
        print(f"[TELEGRAM] Failed: {e}")
        return False

def send_sms(phone_number: str, text: str):
    if not SMS_API_URL or not SMS_API_KEY:
        print("[SMS] Skipped: SMS provider is not configured")
        return False
    try:
        payload = {
            "to": phone_number,
            "from": SMS_SENDER_ID,
            "message": text[:480],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SMS_API_KEY}",
        }
        response = requests.post(SMS_API_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        print(f"[SMS] Sent to {phone_number}")
        return True
    except Exception as e:
        print(f"[SMS] Failed: {e}")
        return False

def deliver_message(channel: str, chat_id: str, phone_number: Optional[str], text: str):
    normalized = (channel or "telegram").lower()
    if normalized in {"sms", "text", "sms_only"}:
        if phone_number:
            return send_sms(phone_number, text)
        print("[SMS] No phone number supplied; falling back to Telegram")
    return send_telegram(chat_id, text)

def build_offline_summary(state) -> str:
    urgency = state.get("urgency_level", "standard")
    score = state.get("urgency_score", 5)
    facility = state.get("facility_recommendation") or {}
    symptoms = ", ".join(state.get("symptoms", [])[:5]) or "not provided"
    return (
        "OFFLINE SUMMARY\n"
        f"Urgency: {urgency.upper()} ({score}/10)\n"
        f"Symptoms: {symptoms}\n"
        f"Suggested facility: {facility.get('name', 'Nearest clinic/hospital')}\n"
        "Store this summary locally and sync when connectivity returns."
    )

def fallback_research_chunks(state) -> List[str]:
    chunks = [c for c in (state.get("research_chunks") or []) if isinstance(c, str) and c.strip()]
    if chunks:
        return chunks[:10]
    raw = state.get("raw_research") or ""
    if not raw.strip():
        return []
    words = raw.split()
    return [" ".join(words[i:i+180]) for i in range(0, len(words), 160) if words[i:i+180]][:10]


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE NODES
# ══════════════════════════════════════════════════════════════════════════════
def validate_node(state):
    raw = state.get("raw_message", "").strip()
    if not raw or len(raw) < 5:
        return {**state, "status": "error_invalid_input",
                "errors": ["Message too short"],
                "workflow_path": state.get("workflow_path", []) + ["validate"]}
    now = datetime.now(timezone.utc)
    tid = f"MED-{now.strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    print(f"[VALIDATE] Ticket: {tid}")
    return {**state, "ticket_id": tid, "initiated_at": now.isoformat(),
            "status": "validated", "workflow_path": state.get("workflow_path", []) + ["validate"]}

def parse_node(state):
    msg   = state["raw_message"]
    corrected = correct_message_for_parsing(msg)
    mode  = detect_user_mode(corrected)
    language = detect_language(corrected, state.get("preferred_language"))
    info  = extract_symptoms(corrected)
    flags = detect_red_flags(corrected)
    condition_hint = match_common_condition(info["symptoms"], [])
    facility = infer_facility(state.get("facility_location"), None, condition_hint)
    if corrected != msg:
        print(f"[PARSE] Corrected spelling: {msg[:80]} -> {corrected[:80]}")
    print(f"[PARSE] Mode: {mode} | Lang: {language} | Red flags: {len(flags)}")
    return {**state, "corrected_message": corrected, "user_mode": mode, "detected_language": language, "symptoms": info["symptoms"],
            "duration": info["duration"], "age": info["age"],
            "medications": info["medications"], "red_flags": flags,
            "facility_recommendation": facility, "status": "parsed",
            "workflow_path": state.get("workflow_path", []) + ["parse"]}

def emergency_node(state):
    print("[EMERGENCY] Fast-track")
    chat_id = state.get("chat_id", "")
    channel = state.get("channel", "telegram")
    phone_number = state.get("phone_number")
    msg = ("🚨 EMERGENCY - GO TO HOSPITAL IMMEDIATELY\n\n"
           "Call 112 (Nigeria Emergency) or go to nearest A&E NOW.\n"
           "Do NOT drive yourself.\n\n"
           "Emergency Numbers:\n- Nigeria Emergency: 112\n- LASAMBUS Lagos: 08000-432584\n\n"
           "⚠️ DISCLAIMER: This is not a substitute for professional medical care.")
    deliver_message(channel, chat_id, phone_number, msg)
    return {**state, "urgency_level": "emergency", "urgency_score": 10,
            "triage_report": {"emergency_response": msg, "patient_report": msg},
            "report_ready": True, "status": "emergency_complete",
            "workflow_path": state.get("workflow_path", []) + ["emergency"]}

def acknowledge_node(state):
    chat_id = state.get("chat_id", "")
    channel = state.get("channel", "telegram")
    phone_number = state.get("phone_number")
    mode    = state.get("user_mode", "patient")
    ticket  = state.get("ticket_id", "")
    if mode == "clinician":
        ack = f"✅ Clinical Query Received\nRef: {ticket}\nSearching WHO + NHIS + PubMed...\nReady in ~60 seconds."
    else:
        ack = f"✅ Symptoms Received\nRef: {ticket}\nReviewing Nigerian health guidelines...\nReady in ~60 seconds.\n\n⚕️ Not a substitute for professional medical advice."
    deliver_message(channel, chat_id, phone_number, ack)
    print(f"[ACK] {mode.upper()} | {ticket}")
    return {**state, "status": "acknowledged",
            "workflow_path": state.get("workflow_path", []) + ["acknowledge"]}

def research_node(state):
    syms  = state.get("symptoms", [])
    msg   = state.get("raw_message", "")
    errors = list(state.get("errors", []))
    query = ", ".join(syms[:4]) if syms else msg[:150]
    print(f"[RESEARCH] {query[:60]}")
    results = []
    for i, topic in enumerate(RESEARCH_TOPICS, 1):
        prompt = f"Nigeria health. Symptoms: {query}. Research: {topic}. Use WHO and NHIS guidelines."
        print(f"  [{i}/{len(RESEARCH_TOPICS)}] {topic[:45]}")
        for attempt in range(3):
            try:
                resp = openai_client.responses.create(
                    model=RESEARCH_MODEL, tools=[{"type": "web_search_preview"}], input=prompt)
                text = "".join(c.text for block in resp.output
                               if hasattr(block, "content") for c in block.content if hasattr(c, "text"))
                if text.strip():
                    results.append(f"### {topic.upper()}\n{text.strip()}")
                    break
            except Exception as e:
                if attempt == 2: errors.append(f"Topic failed: {e}")
                else: time.sleep(2**attempt)
    articles = search_pubmed(f"{query} treatment Nigeria", 3)
    if articles:
        pb = "PUBMED EVIDENCE\n" + "\n".join([f'{a["title"]} ({a["year"]}) {a["url"]}' for a in articles])
        results.append(pb)
    if not results:
        return {**state, "status": "error_research_failed",
                "errors": errors + ["All research failed"],
                "workflow_path": state.get("workflow_path", []) + ["research"]}
    raw = f"MEDICAL RESEARCH: {query}\n\n" + "\n\n".join(results)
    print(f"[RESEARCH] Done: {len(raw):,} chars")
    return {**state, "raw_research": raw, "status": "research_complete", "errors": errors,
            "workflow_path": state.get("workflow_path", []) + ["research"]}

def embed_node(state):
    research  = state.get("raw_research", "")
    ticket    = state.get("ticket_id", "UNKNOWN")
    syms      = state.get("symptoms", [])
    errors    = list(state.get("errors", []))
    words     = research.split()
    chunks    = [" ".join(words[i:i+200]) for i in range(0, len(words), 170) if words[i:i+200]]
    namespace = re.sub(r"[^a-z0-9-]", "", (syms[0] if syms else "general").lower())[:30]
    namespace = f"{namespace}-{ticket[-8:].lower()}"
    print(f"[EMBED] {len(chunks)} chunks -> {namespace}")
    vectors, ids, valid = [], [], []
    for i in range(0, len(chunks), 10):
        batch = chunks[i:i+10]
        for attempt in range(3):
            try:
                emb = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
                for j, e in enumerate(emb.data):
                    cid = f"{ticket}-c{i+j:04d}"
                    vectors.append({"id": cid, "values": e.embedding,
                        "metadata": {"ticket_id": ticket, "text": batch[j][:1000], "chunk_idx": i+j}})
                    ids.append(cid); valid.append(batch[j])
                break
            except Exception as e:
                if attempt == 2: errors.append(str(e))
                else: time.sleep(2**attempt)
    if not vectors:
        return {**state, "status": "error_embed_failed", "errors": errors,
                "workflow_path": state.get("workflow_path", []) + ["embed"]}
    for i in range(0, len(vectors), 100):
        pinecone_index.upsert(vectors=vectors[i:i+100], namespace=namespace)
    print(f"[EMBED] {len(vectors)} vectors stored")
    return {**state, "research_chunks": valid, "pinecone_ids": ids, "clinical_namespace": namespace,
            "status": "stored", "errors": errors, "workflow_path": state.get("workflow_path", []) + ["embed"]}

def retrieve_node(state):
    syms  = state.get("symptoms", [])
    errors = list(state.get("errors", []))
    query = f"Nigeria clinical guidelines: {', '.join(syms[:4])}"
    print(f"[RETRIEVE] {query[:60]}")
    namespace = state.get("clinical_namespace")
    if not namespace:
        syms = state.get("symptoms", [])
        ticket = state.get("ticket_id", "UNKNOWN")
        namespace = re.sub(r"[^a-z0-9-]", "", (syms[0] if syms else "general").lower())[:30]
        namespace = f"{namespace}-{ticket[-8:].lower()}"
    fallback = fallback_research_chunks(state)
    for attempt in range(3):
        try:
            emb    = openai_client.embeddings.create(model=EMBED_MODEL, input=[query], timeout=30)
            res    = pinecone_index.query(vector=emb.data[0].embedding, top_k=5, include_metadata=True, namespace=namespace)
            chunks = [m["metadata"]["text"] for m in res["matches"] if m.get("metadata", {}).get("text")]
            if not chunks:
                if not fallback:
                    fallback = [query]
                print(f"[RETRIEVE] Pinecone returned no matches; using {len(fallback)} local research chunks")
                return {**state, "retrieved_chunks": fallback[:10], "status": "retrieved_fallback", "errors": errors + ["No Pinecone matches; used local research chunks"],
                        "workflow_path": state.get("workflow_path", []) + ["retrieve"]}
            print(f"[RETRIEVE] {len(chunks)} chunks")
            return {**state, "retrieved_chunks": chunks, "status": "retrieved", "errors": errors,
                    "workflow_path": state.get("workflow_path", []) + ["retrieve"]}
        except Exception as e:
            if not fallback:
                fallback = [query]
            print(f"[RETRIEVE] Pinecone error fallback: {e}")
            return {**state, "retrieved_chunks": fallback[:10], "status": "retrieved_fallback",
                    "errors": errors + [str(e), "Used local research chunks fallback"],
                    "workflow_path": state.get("workflow_path", []) + ["retrieve"]}
            time.sleep(2**attempt)

def rerank_node(state):
    syms   = state.get("symptoms", [])
    chunks = state.get("retrieved_chunks", [])
    errors = list(state.get("errors", []))
    query  = f"Nigeria clinical guidelines treatment for: {', '.join(syms[:4])}"
    print(f"[RERANK] {len(chunks)} chunks")
    for attempt in range(3):
        try:
            r = cohere_client.rerank(model="rerank-english-v3.0", query=query,
                                      documents=chunks, top_n=3, return_documents=True)
            reranked = [x.document.text for x in r.results if x.document]
            print(f"[RERANK] Top-3 selected | Scores: {[round(x.relevance_score,3) for x in r.results]}")
            return {**state, "reranked_chunks": reranked, "status": "reranked", "errors": errors,
                    "workflow_path": state.get("workflow_path", []) + ["rerank"]}
        except Exception as e:
            if attempt == 2:
                return {**state, "reranked_chunks": chunks[:3], "status": "reranked_fallback",
                        "errors": errors + [f"Cohere failed: {e}"],
                        "workflow_path": state.get("workflow_path", []) + ["rerank"]}
            time.sleep(2**attempt)

def triage_node(state):
    syms    = state.get("symptoms", [])
    msg     = state.get("raw_message", "")
    mode    = state.get("user_mode", "patient")
    language = state.get("detected_language") or "english"
    facility = state.get("facility_recommendation") or {}
    chunks  = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    errors  = list(state.get("errors", []))
    context = "\n\n---\n\n".join(chunks)
    if mode == "clinician":
        sys_p = (f"Clinical decision support for Nigerian healthcare. Reply in {language}. "
                 "Return JSON: {\"urgency_level\": \"emergency\"|\"urgent\"|\"standard\"|\"self-care\", "
                 "\"urgency_score\": 1-10, \"differential\": [\"cond1 (ICD-10: X00)\",\"cond2\",\"cond3\"], "
                 "\"recommended_action\": \"action\", \"recommended_tests\": [\"test\"], "
                 "\"drug_recommendations\": [\"drug dose\"], \"nhis_protocol\": \"protocol\", "
                 "\"confidence\": \"low\"|\"medium\"|\"high\"} Return ONLY valid JSON.")
    else:
        sys_p = (f"Medical triage AI for Nigeria patients. Reply in {language}. "
                 "Return JSON: {\"urgency_level\": \"emergency\"|\"urgent\"|\"standard\"|\"self-care\", "
                 "\"urgency_score\": 1-10, \"differential\": [\"cond1\",\"cond2\",\"cond3\"], "
                 "\"recommended_action\": \"action\", \"recommended_tests\": [\"test\"], "
                 "\"self_care_eligible\": true/false, \"facility_needed\": \"home\"|\"clinic\"|\"hospital\"|\"emergency_room\", "
                 "\"confidence\": \"low\"|\"medium\"|\"high\"} Return ONLY valid JSON.")
    user_p = (
        f"Context:\n{context}\n\nSymptoms: {syms}\nMessage: {msg}\nMode: {mode}\n"
        f"Facility context: {facility}\n"
        f"Nigerian condition cues: {NIGERIAN_COMMON_CONDITIONS}"
    )
    print(f"[TRIAGE] Mode: {mode}")
    for attempt in range(3):
        try:
            resp   = openai_client.chat.completions.create(
                model=RESEARCH_MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}])
            parsed = json.loads(resp.choices[0].message.content)
            urgency = parsed.get("urgency_level", "standard")
            score   = int(parsed.get("urgency_score", 5))
            diff    = parsed.get("differential", [])
            rule_fallback = classify_urgency_rules(msg, state.get("red_flags", []), syms)
            if rule_fallback.get("urgency_score", 0) > score:
                urgency = rule_fallback["urgency_level"]
                score = rule_fallback["urgency_score"]
            print(f"[TRIAGE] {urgency.upper()} ({score}/10) | {diff[:2]}")
            return {**state, "urgency_level": urgency, "urgency_score": score,
                    "differential": diff, "status": "triaged", "errors": errors,
                    "triage_report": {"_classification": json.dumps(parsed)},
                    "workflow_path": state.get("workflow_path", []) + ["triage"]}
        except Exception as e:
            if attempt == 2:
                rule_fallback = classify_urgency_rules(msg, state.get("red_flags", []), syms)
                return {**state, "status": "error_triage_failed",
                        "urgency_level": rule_fallback["urgency_level"],
                        "urgency_score": rule_fallback["urgency_score"],
                        "errors": errors + [str(e), "Used rule-based urgency fallback"],
                        "workflow_path": state.get("workflow_path", []) + ["triage"]}
            time.sleep(2**attempt)

def report_node(state):
    urgency  = state.get("urgency_level", "standard")
    score    = state.get("urgency_score", 5)
    diff     = state.get("differential", [])
    syms     = state.get("symptoms", [])
    mode     = state.get("user_mode", "patient")
    language = state.get("detected_language") or "english"
    ticket   = state.get("ticket_id", "")
    chunks   = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    errors   = list(state.get("errors", []))
    context  = "\n\n".join(chunks[:3])
    action   = URGENCY_ACTIONS.get(urgency, "See a doctor.")
    facility = state.get("facility_recommendation") or infer_facility(
        state.get("facility_location"), urgency, match_common_condition(syms, diff)
    )
    offline_pack = build_offline_summary(state)
    print(f"[REPORT] {mode} | {urgency.upper()}")
    if mode == "clinician":
        sys_p = (f"Clinical decision support Nigeria. Generate structured clinical report in {language} with ICD-10 codes, "
                 "tests, first-line drugs and doses, NHIS protocol. Under 300 words.")
        usr_p = f"Context:\n{context}\n\nPresentation: {syms}\n\nFacility: {facility}\n\nGenerate clinical triage report."
    else:
        sys_p = (f"Patient health assistant Nigeria. Write clear 150-word triage report in {language}. "
                 "No medical jargon. Base on provided context. Do not add disclaimer yet.")
        usr_p = f"Context:\n{context}\n\nSymptoms: {syms}\nUrgency: {urgency}\nLikely: {diff}\nFacility: {facility}"
    for attempt in range(3):
        try:
            resp = openai_client.chat.completions.create(
                model=REPORT_MODEL, temperature=0.2,
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}])
            body = resp.choices[0].message.content.strip()
            full = (f"TRIAGE ASSESSMENT\nRef: {ticket}\n"
                    f"Urgency: {urgency.upper()} ({score}/10)\n\n"
                    f"ACTION: {action}\n\n{body}\n\n"
                    f"FACILITY RECOMMENDATION:\n- {facility.get('name', 'Nearest appropriate facility')}\n"
                    f"- Level: {facility.get('level', 'unknown')}\n"
                    f"- Reason: {facility.get('reason', 'Based on symptoms and local care needs')}\n\n"
                    f"OFFLINE CAPABILITY:\n{offline_pack}\n\n"
                    f"DISCLAIMER: This is not a substitute for professional medical advice. "
                    f"Always consult a qualified healthcare provider.")
            if state.get("offline_mode"):
                full += "\n\nOFFLINE MODE: This response is formatted for local caching and later sync."
            existing = state.get("triage_report") or {}
            key = "clinician_report" if mode == "clinician" else "patient_report"
            existing[key] = full
            print(f"[REPORT] Generated ({len(full)} chars)")
            return {**state, "triage_report": existing, "status": "report_generated",
                    "errors": errors, "workflow_path": state.get("workflow_path", []) + ["report"]}
        except Exception as e:
            if attempt == 2:
                return {**state, "status": "error_report_failed",
                        "errors": errors + [str(e)],
                        "workflow_path": state.get("workflow_path", []) + ["report"]}
            time.sleep(2**attempt)

def deliver_node(state):
    mode      = state.get("user_mode", "patient")
    urgency   = state.get("urgency_level", "standard")
    report    = state.get("triage_report") or {}
    chat_id   = state.get("chat_id", "")
    phone_number = state.get("phone_number")
    channel  = state.get("channel", "telegram")
    errors    = list(state.get("errors", []))
    key       = "clinician_report" if mode == "clinician" else "patient_report"
    body      = report.get(key, report.get("emergency_response", "Report unavailable"))
    if urgency == "self-care":
        nutrition = get_nutrition_advice(state.get("symptoms", []), state.get("differential", []))
        body     += f"\n\nNIGERIAN NUTRITION ADVICE:\n{nutrition}"
        body     += "\n\nPROGRESS CHECK: You will receive a check-in message in 24 hours."
    delivered = deliver_message(channel, chat_id, phone_number, body)
    print(f"[DELIVER] Report sent ({len(body)} chars) | delivered={delivered}")
    return {**state, "report_ready": True, "status": "complete", "errors": errors,
            "workflow_path": state.get("workflow_path", []) + ["deliver"]}

def error_handler_node(state):
    print(f"\nPIPELINE ERROR: {state.get('status')}")
    for e in state.get("errors", []): print(f"  {e}")
    chat_id = state.get("chat_id", "")
    channel = state.get("channel", "telegram")
    phone_number = state.get("phone_number")
    err_msg = (f"Sorry, we encountered an error processing your request.\n"
               f"Ref: {state.get('ticket_id', 'N/A')}\n"
               f"Please try again or contact support.\n\n"
               f"DISCLAIMER: Always consult a qualified healthcare provider.")
    deliver_message(channel, chat_id, phone_number, err_msg)
    return {**state, "status": "failed",
            "workflow_path": state.get("workflow_path", []) + ["error_handler"]}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def route(state) -> str:
    return "error_handler" if "error" in state.get("status", "") else "continue"

def route_after_parse(state) -> str:
    return "emergency" if state.get("red_flags") else "acknowledge"

def build_pipeline():
    g = StateGraph(MedicalAgentState)
    for name, fn in [
        ("validate", validate_node), ("parse", parse_node),
        ("emergency", emergency_node), ("acknowledge", acknowledge_node),
        ("research", research_node), ("embed", embed_node),
        ("retrieve", retrieve_node), ("rerank", rerank_node),
        ("triage", triage_node), ("report", report_node),
        ("deliver", deliver_node), ("error_handler", error_handler_node),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("validate")
    g.add_conditional_edges("validate", route, {"continue": "parse", "error_handler": "error_handler"})
    g.add_conditional_edges("parse", route_after_parse, {"emergency": "emergency", "acknowledge": "acknowledge"})
    g.add_edge("emergency", "deliver")
    g.add_conditional_edges("acknowledge", route, {"continue": "research", "error_handler": "error_handler"})
    for step, nxt in [("research","embed"),("embed","retrieve"),("retrieve","rerank"),
                       ("rerank","triage"),("triage","report"),("report","deliver")]:
        g.add_conditional_edges(step, route, {"continue": nxt, "error_handler": "error_handler"})
    g.add_conditional_edges("deliver", route, {"continue": END, "error_handler": "error_handler"})
    g.add_edge("error_handler", END)
    return g.compile()

print("Building pipeline...")
pipeline = build_pipeline()
print("Medical Triage Pipeline ready - 11 nodes")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Medical Triage Agent", version="1.0.0",
              description="Autonomous Medical Research & Patient Triage Assistant — Nigeria")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TriageRequest(BaseModel):
    message:  str
    channel:  str = "telegram"
    chat_id:  str = ""
    phone_number: Optional[str] = None
    preferred_language: Optional[str] = None
    facility_location: Optional[str] = None
    offline_mode: bool = False

class TriageResponse(BaseModel):
    ticket_id:      str
    status:         str
    user_mode:      str
    detected_language: Optional[str]
    urgency_level:  Optional[str]
    urgency_score:  Optional[int]
    differential:   Optional[List[str]]
    report_ready:   bool
    triage_report:  Optional[Dict[str, str]]
    nutrition_advice: Optional[str]
    facility_recommendation: Optional[Dict[str, str]]
    offline_mode: bool
    errors:         List[str]
    workflow_path:  List[str]


@app.get("/health")
def health():
    return {"status": "ok", "service": "Medical Triage Agent", "pipeline": "ready"}

@app.post("/triage", response_model=TriageResponse)
def run_triage(request: TriageRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    print(f"\n{'='*55}")
    print(f"  NEW TRIAGE: {request.message[:60]}")
    print(f"  Channel   : {request.channel} | Chat: {request.chat_id}")
    print(f"{'='*55}")

    initial_state = MedicalAgentState(
        session_id=str(uuid.uuid4()), ticket_id="", initiated_at="",
        channel=request.channel, chat_id=request.chat_id,
        phone_number=request.phone_number,
        user_mode="patient", raw_message=request.message, corrected_message=None,
        preferred_language=request.preferred_language,
        detected_language=None,
        facility_location=request.facility_location,
        facility_recommendation=None,
        offline_mode=request.offline_mode,
        symptoms=[], duration=None, age=None, medications=[],
        status="pending", errors=[],
        raw_research=None, research_chunks=None, pinecone_ids=None,
        retrieved_chunks=None, reranked_chunks=None,
        urgency_level=None, urgency_score=None,
        differential=None, red_flags=None, retry_count=0,
        triage_report=None, nutrition_advice=None,
        home_remedies=None, follow_up_sent=False,
        report_ready=False, workflow_path=[]
    )

    result = pipeline.invoke(initial_state)
    print(f"\n✅ COMPLETE: {result['status']} | {result.get('urgency_level')} | {result.get('urgency_score')}/10")

    return TriageResponse(
        ticket_id       = result.get("ticket_id", ""),
        status          = result.get("status", ""),
        user_mode       = result.get("user_mode", "patient"),
        detected_language = result.get("detected_language"),
        urgency_level   = result.get("urgency_level"),
        urgency_score   = result.get("urgency_score"),
        differential    = result.get("differential"),
        report_ready    = result.get("report_ready", False),
        triage_report   = result.get("triage_report"),
        nutrition_advice= result.get("nutrition_advice"),
        facility_recommendation = result.get("facility_recommendation"),
        offline_mode    = result.get("offline_mode", False),
        errors          = result.get("errors", []),
        workflow_path   = result.get("workflow_path", [])
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    print("\n" + "="*55)
    print("  🏥 Medical Triage Agent Server")
    print(f"  URL   : http://localhost:{port}")
    print(f"  Docs  : http://localhost:{port}/docs")
    print(f"  Health: http://localhost:{port}/health")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
