# IntelliFraud Copilot (Agentic RAG MVP)

A runnable, end-to-end **agentic RAG** backend that simulates a fraud-investigation copilot:
- **Planner Agent** decides which tools/agents to invoke
- **Transaction Data Agent** queries a transaction store (SQLite)
- **Policy RAG Agent** retrieves relevant policy snippets (FAISS vector index)
- **Fraud Reasoning Agent** correlates signals (DeepSeek by default, falls back to OpenAI)
- **Explanation Agent** produces a grounded analyst-friendly summary with citations

> Security/auth/PII hardening is intentionally minimal in this MVP.

## Quickstart

### 1) Create `.env`
Copy:
```bash
cp .env.example .env
```

Fill in:
- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY` (optional; if missing we fall back to OpenAI for reasoning too)

### 2) Install & run
```bash
pip install -r requirements.txt
python -m app.db.seed_data
uvicorn app.main:app --reload
```

### 3) Call the API
```bash
curl -X POST http://127.0.0.1:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"query":"Investigate transaction 89342 and cite the relevant policy"}'
```

## Endpoints
- `GET /health`
- `POST /investigate` — primary endpoint
- `POST /rag/reindex` — rebuild policy index

## Project Layout
See `app/` for code, `data/` for sample datasets.
