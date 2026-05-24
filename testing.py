import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

payload = {
    "message": "I have fever and headache for 3 days",
    "channel": "telegram",
    "chat_id": "8138298582",
    "phone_number": None,
    "preferred_language": None,
    "facility_location": None,
    "offline_mode": False,
}


def get_target() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1].strip().lower()
    return os.getenv("TEST_TARGET", "n8n").strip().lower()


def get_url(target: str) -> str:
    if target == "render":
        return os.getenv("TRIAGE_API_URL", "").strip()
    return os.getenv("N8N_WEBHOOK_URL", "").strip()


def main() -> int:
    target = get_target()
    if target not in {"n8n", "render"}:
        print("Usage: python testing.py [n8n|render]")
        return 1

    url = get_url(target)
    if not url:
        if target == "n8n":
            print("Missing N8N_WEBHOOK_URL in .env")
            print("Example:")
            print("  N8N_WEBHOOK_URL=https://adetu-o.n8n.irn.hk/webhook-test/medical-triage")
        else:
            print("Missing TRIAGE_API_URL in .env")
            print("Example:")
            print("  TRIAGE_API_URL=https://medical-triage-j8fm.onrender.com/triage")
        return 1

    print(f"Target: {target}")
    print(f"URL: {url}")

    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"Status: {response.status_code}")
        print("Body:")
        try:
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        except ValueError:
            print(response.text)
        response.raise_for_status()
        return 0
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
