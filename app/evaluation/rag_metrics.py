from __future__ import annotations
from typing import Dict, Any, List
import re

def keyword_coverage(answer: str, retrieved: List[Dict[str, Any]]) -> float:
    """
    Tiny heuristic metric: how many content keywords from retrieved chunks appear in answer.
    This is NOT a substitute for RAGAS; it's an MVP sanity check.
    """
    ctx = " ".join(r.get("text","") for r in retrieved).lower()
    ans = (answer or "").lower()
    tokens = [t for t in re.findall(r"[a-z]{5,}", ctx)][:50]
    if not tokens:
        return 0.0
    hit = sum(1 for t in set(tokens) if t in ans)
    return hit / max(1, len(set(tokens)))

def simple_report(explained: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = (explained.get("explanation") or {}).get("summary","")
    return {"keyword_coverage": keyword_coverage(summary, retrieved)}
