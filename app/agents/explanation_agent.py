from __future__ import annotations
import json
from typing import Dict, Any
from app.models.openai_client import OpenAIChatClient
from app.rag.grounding import grounded_payload
from app.utils.logger import get_logger

logger = get_logger(__name__)

EXPLAIN_PROMPT = """You are an Explanation Agent for fraud analysts.

You will be given structured reasoning JSON with citations context.
Write a concise explanation for an analyst.

Rules:
- Use only the provided evidence.
- Avoid speculation.
- Include a 'Key Evidence' bullet list referencing citations by (doc_id, section).
- Output JSON ONLY with:
  - summary: str
  - key_evidence: list[str]
  - risk_score: number (0-100)
  - confidence: number (0-1)
"""

class ExplanationAgent:
    def __init__(self):
        self.llm = OpenAIChatClient()

    async def run(self, reasoning: Dict[str, Any], retrieved_chunks: list) -> Dict[str, Any]:
        # Add citations to reasoning so the LLM can reference them.
        packed = {
            "reasoning": reasoning,
            "citations_available": [
                {"doc_id": c["doc_id"], "section": c["section"], "chunk_id": c.get("chunk_id")} for c in retrieved_chunks
            ],
        }
        raw = await self.llm.complete(EXPLAIN_PROMPT + "\n\n" + json.dumps(packed, ensure_ascii=False), temperature=0.2)

        try:
            out = json.loads(raw)
        except Exception:
            s = raw.find("{"); e = raw.rfind("}")
            out = json.loads(raw[s:e+1])

        # Ensure grounding payload for the API response includes retrieved context
        return {
            "explanation": out,
            "grounding": grounded_payload(
                answer=out.get("summary",""),
                retrieved=retrieved_chunks,
                confidence=float(out.get("confidence", 0.0) or 0.0),
            )
        }
