# Lab Summary

## Project Title
Medical Triage Agent

## Goal
Build an autonomous triage assistant for Nigerian healthcare workflows that can assess symptoms, classify urgency, and return a structured response for patients or clinicians.

## What It Does
- Detects whether the user is in patient mode or clinician mode.
- Extracts symptoms, duration, age, and medication mentions from free-text input.
- Screens for red-flag symptoms and fast-tracks emergencies.
- Uses research-backed retrieval from web search, PubMed, Pinecone, and Cohere reranking.
- Generates triage outputs with urgency level, differential diagnosis, action steps, and safety disclaimer.
- Supports Nigerian context, including common conditions, nutrition advice, facility suggestions, and emergency contacts.
- Supports multi-language handling, including English, Hausa, Igbo, and Yoruba.
- Includes SMS workflow support and offline-summary considerations.

## Core Tech Stack
- Python
- FastAPI
- LangGraph
- OpenAI API
- Pinecone
- Cohere
- PubMed E-utilities
- Telegram delivery

## Key API Endpoint
- `POST /triage`

Example request:
```json
{
  "message": "I have fever and headache for 3 days",
  "channel": "telegram",
  "chat_id": "8138298582"
}
```

## Output
The service returns:
- `ticket_id`
- `status`
- `user_mode`
- `urgency_level`
- `urgency_score`
- `differential`
- `triage_report`
- `nutrition_advice`
- `facility_recommendation`
- `workflow_path`
- `errors`

## Main Workflow
1. Validate the message.
2. Parse symptoms, language, and red flags.
3. Emergency fast-track if needed.
4. Acknowledge the request.
5. Research the condition context.
6. Embed and retrieve clinical evidence.
7. Rerank the evidence.
8. Classify urgency.
9. Generate the report.
10. Deliver the result by Telegram or SMS.

## Nigerian Context Included
- Common conditions such as malaria, typhoid, hypertension, diabetes, gastroenteritis, and asthma.
- Nigerian nutrition guidance.
- Facility guidance for clinic, hospital, and emergency care.
- Emergency phone number references.

## Safety Notes
- This is decision support, not a substitute for a clinician.
- Emergency symptoms always bypass the normal workflow.
- Outputs should be reviewed carefully in real-world deployments.

## Project Files
- `medical_agent_server.py` - main FastAPI triage server
- `sprint1_medical.ipynb` to `sprint5_medical.ipynb` - development notebooks
- `sprint*_medical_output.json` - generated outputs from sprint runs

