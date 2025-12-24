from __future__ import annotations
from typing import Dict, List, Any

def make_citations(retrieved: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cites = []
    for r in retrieved:
        cites.append({
            "doc_id": r.get("doc_id", "unknown"),
            "section": r.get("section", "unknown"),
            "chunk_id": str(r.get("chunk_id", "")),
        })
    return cites

def grounded_payload(answer: str, retrieved: List[Dict[str, Any]], confidence: float) -> Dict[str, Any]:
    return {
        "answer": answer,
        "confidence": confidence,
        "citations": make_citations(retrieved),
        "retrieved_context": [
            {"doc_id": r["doc_id"], "section": r["section"], "text": r["text"][:800]} for r in retrieved
        ],
    }
