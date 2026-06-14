"""Quick API smoke test (run while uvicorn is up)."""
import json
import os
import sys
import urllib.request

_PORT = os.environ.get("SA_AQG_API_PORT", "8000")
BASE = f"http://127.0.0.1:{_PORT}"


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}") as resp:
        return json.loads(resp.read())


def post(path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def upload_cv_text(text: str) -> dict:
    import uuid
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="cv.txt"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
        f"{text}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/api/interview/upload-cv",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


if __name__ == "__main__":
    print("health:", get("/api/health"))
    upload = upload_cv_text("5 years Python and Kubernetes experience.")
    print("upload session:", upload.get("cv_session_id", "")[:8], "...")
    status, result = post(
        "/api/interview/generate",
        {
            "specialization": "devops",
            "experience_level": "senior",
            "mode": "mixed",
            "cv_session_id": upload["cv_session_id"],
            "num_questions": 1,
        },
    )
    print("generate status:", status)
    print("questions:", len(result.get("questions", [])))
    if result.get("questions"):
        print("first question:", result["questions"][0]["question"][:120])
    sys.exit(0 if status == 200 and result.get("questions") else 1)
