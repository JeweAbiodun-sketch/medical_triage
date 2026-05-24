# Medical Triage Agent

Autonomous triage assistant for Nigerian healthcare workflows.

## Repository
- GitHub: [JeweAbiodun-sketch/medical_triage](https://github.com/JeweAbiodun-sketch/medical_triage)

## Overview
This project combines symptom parsing, emergency screening, clinical research retrieval, urgency classification, and report generation into one FastAPI service.

It is designed for:
- Patients who need plain-language triage guidance
- Patients who need an interactive chatbot that asks follow-up questions
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
- LLM-driven chatbot intake with per-chat memory and a triage tool handoff

## Main File
- [medical_agent_server.py](./medical_agent_server.py)

## n8n Workflow
- [n8n/medical_triage_intake.json](./n8n/medical_triage_intake.json)
- [n8n/telegram_chatbot_intake.json](./n8n/telegram_chatbot_intake.json)

Import this workflow into n8n to receive a triage request, normalize the payload, call the FastAPI service, and return the structured triage response.
Use the Telegram workflow for interactive intake that forwards Telegram messages to the chatbot endpoint, keeps memory in the backend, and calls the triage pipeline when the model decides it has enough information.

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

Default local port:
- `8001`

If that port is already in use, set a different one before starting:
```bash
PORT=8002 python medical_agent_server.py
```

The service will then start on:
- `http://localhost:8001` by default
- `http://localhost:<PORT>` when `PORT` is set

Helpful endpoints:
- `GET /health`
- `POST /triage`
- `POST /chatbot/telegram`

## Example Request
```bash
curl -X POST http://localhost:8001/triage ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"I have fever and headache for 3 days\",\"channel\":\"telegram\",\"chat_id\":\"8138298582\"}"
```

## Chatbot Request
```bash
curl -X POST http://localhost:8001/chatbot/telegram ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"I have fever and headache for 3 days\",\"channel\":\"telegram\",\"chat_id\":\"8138298582\",\"user_name\":\"Abiodun\"}"
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
- `n8n/medical_triage_intake.json` - n8n intake workflow export
- `AGENTS_medical.md` - workflow and agent design notes
- `sprint1_medical.ipynb` to `sprint5_medical.ipynb` - sprint notebooks
- `sprint*_medical_output.json` - output artifacts

## n8n Setup
1. Start the FastAPI server locally.
2. Set `TRIAGE_API_URL` in n8n if needed.
3. Import `n8n/medical_triage_intake.json` for webhook intake or `n8n/telegram_chatbot_intake.json` for the interactive chatbot.
4. The Telegram chatbot workflow now calls `POST /chatbot/telegram` on the backend, which keeps memory and decides when to hand off to `/triage`.
5. Activate the workflow and send a Telegram message to your bot.

## Testing Helper
Use `testing.py` to smoke-test either endpoint:

Test the n8n webhook:
```bash
python testing.py n8n
```

Test the Render API:
```bash
python testing.py render
```

Test the chatbot endpoint directly:
```bash
python testing.py chatbot
```

Optional `.env` values:
```bash
N8N_WEBHOOK_URL=https://adetu-o.n8n.irn.hk/webhook/medical-triage
TRIAGE_API_URL=https://medical-triage-j8fm.onrender.com/triage
CHATBOT_API_URL=https://medical-triage-j8fm.onrender.com/chatbot/telegram
TEST_TARGET=n8n
```

## Render Deployment
Live API base URL:
- `https://medical-triage-j8fm.onrender.com`

Render sets `PORT` automatically, and `medical_agent_server.py` now reads it.

Use this in n8n as:
- `https://medical-triage-j8fm.onrender.com/triage`

If you want to override the default in n8n, set:
```bash
TRIAGE_API_URL=https://medical-triage-j8fm.onrender.com/triage
```

## Notes
- Emergency symptoms bypass the standard pipeline.
- The SMS layer depends on an external provider.
- Facility recommendations are currently based on a starter directory and can be expanded with a live dataset.

## Safety Disclaimer
This project is for triage assistance and education. It does not replace professional medical advice, diagnosis, or treatment.
