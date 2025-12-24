from __future__ import annotations
from typing import Dict, Any
from app.db.vector_store import get_policy_index
from app.utils.config import settings

class PolicyRAGAgent:
    """
    Retrieves policy snippets relevant to the investigation question.
    """
    def run(self, query: str) -> Dict[str, Any]:
        idx = get_policy_index()
        retrieved = idx.query(query, top_k=settings.top_k, rerank_k=settings.rerank_k)
        if not retrieved:
            return {"status": "INSUFFICIENT_POLICY_CONTEXT", "retrieved": []}
        return {"status": "OK", "retrieved": retrieved}
