# Medical Triage Agent

Autonomous triage assistant for Nigerian healthcare workflows.

## Repository
- GitHub: [JeweAbiodun-sketch/medical_triage](https://github.com/JeweAbiodun-sketch/medical_triage)

## Overview
This project combines symptom parsing, emergency screening, clinical research retrieval, urgency classification, and report generation into one FastAPI service.

It is designed for:
- Patients who need plain-language triage guidance
- Clinicians who need structured clinical support
- Nigerian healthcare contexts, including local conditions, emergency contacts, and facility suggestions

## Features
- Multi-language support: English, Hausa, Igbo, Yoruba
- Nigerian medical context and common conditions
- Facility location integration
- SMS workflow optimization
- Offline capability considerations
- Urgency classification system
- Emergency red-flag fast track
- Telegram delivery support

## Main File
- [medical_agent_server.py](./medical_agent_server.py)

## Run the Server
Create and activate a virtual environment if you do not already have one:
```bash
python -m venv .venv
```

Install dependencies first:
```bash
pip install -r requirements.txt
```

Then start the server:
```bash
python medical_agent_server.py
```

The service starts on:
- `http://localhost:8001`

Helpful endpoints:
- `GET /health`
- `POST /triage`

## Example Request
```bash
curl -X POST http://localhost:8001/triage ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"I have fever and headache for 3 days\",\"channel\":\"telegram\",\"chat_id\":\"8138298582\"}"
```

## Request Body
```json
{
  "message": "I have fever and headache for 3 days",
  "channel": "telegram",
  "chat_id": "8138298582",
  "phone_number": null,
  "preferred_language": null,
  "facility_location": null,
  "offline_mode": false
}
```

## Response Includes
- Ticket ID
- Status
- User mode
- Detected language
- Urgency level and score
- Differential diagnosis
- Report content
- Nutrition advice
- Facility recommendation
- Workflow path
- Errors

## Environment Variables
Required:
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `COHERE_API_KEY`

Optional:
- `TELEGRAM_BOT_TOKEN`
- `SMS_API_URL`
- `SMS_API_KEY`
- `SMS_SENDER_ID`

## Project Structure
- `medical_agent_server.py` - API server and workflow
- `AGENTS_medical.md` - workflow and agent design notes
- `sprint1_medical.ipynb` to `sprint5_medical.ipynb` - sprint notebooks
- `sprint*_medical_output.json` - output artifacts

## Notes
- Emergency symptoms bypass the standard pipeline.
- The SMS layer depends on an external provider.
- Facility recommendations are currently based on a starter directory and can be expanded with a live dataset.

## Safety Disclaimer
This project is for triage assistance and education. It does not replace professional medical advice, diagnosis, or treatment.
