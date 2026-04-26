# Deploy Gemini-API to Vercel

This repository is a Python SDK by default. To deploy it as a web service, this project now includes a FastAPI serverless wrapper in `api/index.py`.

## 1) Prerequisites

- Python 3.10+
- Node.js 18+
- A Vercel account
- Gemini cookies:
  - `__Secure-1PSID`
  - `__Secure-1PSIDTS` (optional for some accounts)

## 2) Local run (recommended before deploy)

```bash
pip install -r requirements.txt
uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload
```

Set environment variables before running:

```bash
GEMINI_SECURE_1PSID=your_cookie
GEMINI_SECURE_1PSIDTS=your_cookie
GEMINI_DEFAULT_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT_SECONDS=180
```

Quick test:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello from local API"}'
```

## 3) Deploy with Vercel CLI

Install Vercel CLI:

```bash
npm i -g vercel
```

Login and deploy:

```bash
vercel
vercel --prod
```

Set environment variables in Vercel project settings (or CLI):

- `GEMINI_SECURE_1PSID` (required)
- `GEMINI_SECURE_1PSIDTS` (optional)
- `GEMINI_DEFAULT_MODEL` (optional)
- `GEMINI_PROXY` (optional)
- `GEMINI_TIMEOUT_SECONDS` (optional)

## 4) API endpoints

- `GET /` basic info
- `GET /health` health check
- `GET /models` list models for the authenticated account
- `POST /chat` generate one response

`POST /chat` payload example:

```json
{
  "prompt": "Write a short haiku about spring.",
  "model": "gemini-2.5-flash",
  "temporary": false,
  "deep_research": false
}
```
