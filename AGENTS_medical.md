# AGENTS.md
## Autonomous Medical Research & Patient Triage Assistant

**Project:** Module — Individual Project
**Geography:** Nigeria (NHIS Guidelines + WHO Protocols)
**Channels:** Telegram + WhatsApp
**Version:** 1.0

---

## Agent Name & Purpose

**Name:** MedicalTriageAgent
**Purpose:** Autonomously research patient symptoms against WHO and NHIS Nigeria guidelines, PubMed peer-reviewed evidence, and deliver a structured triage assessment with urgency level, differential diagnosis, and Nigerian-context nutrition advice — via Telegram or WhatsApp, in under 3 minutes.

---

## Tools Available

| Tool | Node | When to Use | API |
|------|------|-------------|-----|
| OpenAI Web Search | `research_node` | Always — for 8 clinical research topics | `openai.responses.create` with `web_search_preview` |
| PubMed API | `research_node` | Always — peer-reviewed evidence | NCBI E-utilities (free, no key) |
| OpenAI Embeddings | `embed_node` | After research — vectorise chunks | `openai.embeddings.create` (text-embedding-3-small) |
| Pinecone Upsert | `embed_node` | After embedding — store clinical vectors | `pinecone_index.upsert` |
| Pinecone Query | `retrieve_node` | After storage — retrieve clinical evidence | `pinecone_index.query` |
| Cohere Rerank | `rerank_node` | After retrieval — clinical precision filter | `cohere_client.rerank` (rerank-english-v3.0) |
| OpenAI Chat | `triage_node` | After reranking — classify urgency | `openai.chat.completions.create` (gpt-4o-mini) |
| OpenAI Chat | `report_node` | After triage — generate report | `openai.chat.completions.create` (gpt-4o) |
| Telegram API | `deliver_node` | After report — send to patient/clinician | `POST /sendMessage` |

---

## Research Workflow (Step-by-Step)

```
1. VALIDATE    → Sanitise message, generate ticket ID (MED-YYYYMMDD-XXXXXXXX)
2. PARSE       → Detect mode (patient/clinician), extract symptoms, screen red flags
3. EMERGENCY   → If red flags: fast-track emergency response, skip full pipeline
4. ACKNOWLEDGE → Send confirmation to patient/clinician via Telegram/WhatsApp
5. RESEARCH    → OpenAI web search × 8 clinical topics + PubMed API
6. EMBED       → Chunk (200 words/30 overlap) → embed → upsert to Pinecone
7. RETRIEVE    → Embed query → Pinecone top-10 similarity search
8. RERANK      → Cohere cross-encoder → top-3 most clinically relevant chunks
9. TRIAGE      → LangGraph agent classifies urgency, differential, recommended action
10. REPORT     → GPT-4o generates structured report (patient or clinician mode)
11. DELIVER    → Send report + nutrition advice via Telegram/WhatsApp
```

---

## User Modes

### Patient Mode
Triggered by: default (no clinician keywords detected)
Output: Plain language, no jargon, urgency colour, action, Nigerian food advice

### Clinician Mode
Triggered by: "i am a doctor", "my patient", "presenting with", "differential", etc.
Output: ICD-10 codes, drug recommendations with doses, contraindications, NHIS protocol, PubMed citation

---

## Urgency Levels

| Level | Emoji | Score | Action |
|-------|-------|-------|--------|
| emergency | 🔴 | 9-10 | Go to A&E immediately. Call 112. |
| urgent | 🟠 | 6-8 | See doctor within 24 hours |
| standard | 🟡 | 3-5 | Book clinic appointment this week |
| self-care | 🟢 | 1-2 | Manage at home + nutrition advice + 24hr follow-up |

---

## Red Flag Symptoms (Emergency Fast-Track)

| Category | Keywords |
|----------|----------|
| cardiac | chest pain, cannot breathe, shortness of breath |
| neurological | seizure, convulsion, fits, unconscious, stroke |
| haemorrhage | vomiting blood, blood in stool, bleeding heavily |
| sepsis | temperature 40, not responding, shaking badly |

---

## Output Format

### Patient Report Structure
```
TRIAGE ASSESSMENT
Ref: MED-YYYYMMDD-XXXXXXXX
Urgency: URGENT (7/10)

ACTION: See a doctor within 24 hours.

[150-word plain language assessment]
- Likely causes (simple terms)
- Tests to get if seeing doctor
- Warning signs to watch
- General care advice

DISCLAIMER: This is not a substitute for professional medical advice.
Always consult a qualified healthcare provider.
```

### Clinician Report Structure
```
CLINICAL TRIAGE REPORT
Ref: MED-YYYYMMDD-XXXXXXXX
Urgency: URGENT (7/10)

[Clinical report including:]
- Top 3 differential diagnoses with ICD-10 codes
- Recommended investigations (with rationale)
- First-line treatment (drugs, doses, routes)
- NHIS Nigeria protocol reference
- PubMed citation from evidence context
- Referral criteria

DISCLAIMER: Clinical judgment of treating physician supersedes this report.
```

### Self-Care Addition
```
NIGERIAN NUTRITION ADVICE:
FOODS TO EAT:
  - [Nigerian food 1]
  - [Nigerian food 2]
FOODS TO AVOID:
  - [Food to avoid]
TIP: [Evidence-based tip]

PROGRESS CHECK: You will receive a check-in message in 24 hours.
```

---

## Error Behaviour

| Error Type | Behaviour |
|-----------|-----------|
| Message < 5 chars | Pipeline stops at validate — error message returned |
| Red flags detected | Emergency fast-track — bypass full pipeline, immediate response |
| OpenAI API failure | 3 retries with exponential backoff (1s, 2s, 4s) |
| PubMed failure | Logged, continue with web search data only |
| Pinecone failure | 3 retries — if all fail, pipeline stops with error |
| Cohere failure | Falls back to top-3 Pinecone similarity results |
| Report generation failure | Section marked failed — pipeline continues |
| Telegram delivery failure | Logged in errors — report data still returned in API response |
| Any node failure | `error_handler_node` fires — sends error message to patient via Telegram |

---

## Safety Requirements (Non-Negotiable)

1. **Every response** must include the safety disclaimer
2. **Emergency symptoms** must trigger fast-track — never wait for full pipeline
3. **No fabrication** — all recommendations grounded in retrieved clinical evidence
4. **Mode detection** — clinicians get clinical output, patients get plain language
5. **Nigeria-specific** — always reference NHIS and WHO Nigeria guidelines where available

---

## Skills Reference

| Skill File | Covers |
|-----------|--------|
| `skills/research.md` | Web search prompts, PubMed integration, chunking strategy |
| `skills/triage.md` | Urgency scoring rubric, red flag detection, mode classification |
| `skills/report_gen.md` | Patient vs clinician prompts, nutrition database, disclaimer format |
| `skills/safety.md` | Emergency fast-track rules, disclaimer requirements, red flag list |

---

## Environment Variables Required

```bash
OPENAI_API_KEY    = sk-...
PINECONE_API_KEY  = pcsk_...
COHERE_API_KEY    = ...
TELEGRAM_BOT_TOKEN = ...   # Optional but recommended
TELEGRAM_CHAT_ID   = ...   # Optional
```

---

## Notebook Structure

| Notebook | Sprint | Covers |
|---------|--------|--------|
| `sprint1_medical.ipynb` | Day 1 | MedicalAgentState, validation, mode detection, red flags |
| `sprint2_medical.ipynb` | Day 2 | PubMed API, web search, Pinecone storage |
| `sprint3_medical.ipynb` | Day 3 | RAG retrieval, Cohere reranking, LangGraph triage |
| `sprint4_medical.ipynb` | Day 4 | Report generation, nutrition advice, Telegram delivery |
| `sprint5_medical.ipynb` | Day 5 | Full pipeline, QA × 3 conditions, documentation |

---

## API Endpoint

```
POST http://localhost:8001/triage

Request:
{
  "message": "I have fever and headache for 3 days",
  "channel": "telegram",
  "chat_id": "8138298582"
}

Response:
{
  "ticket_id": "MED-20260523-A1B2C3D4",
  "status": "complete",
  "user_mode": "patient",
  "urgency_level": "urgent",
  "urgency_score": 7,
  "differential": ["Malaria", "Typhoid fever", "Viral fever"],
  "report_ready": true,
  "triage_report": { "patient_report": "..." },
  "nutrition_advice": "FOODS TO EAT:...",
  "errors": [],
  "workflow_path": ["validate","parse","acknowledge","research",...]
}
```
