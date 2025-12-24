from __future__ import annotations
import json
from typing import Dict, Any
from app.models.deepseek_client import DeepSeekClient
from app.models.openai_client import OpenAIChatClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

REASONING_PROMPT = """You are a Fraud Reasoning Agent.

Inputs:
- Transaction signals (JSON)
- Retrieved policy excerpts (JSON list)
- Historical fraud patterns (JSON list)

Task:
1) Identify suspicious signals.
2) Match to likely fraud typologies from the historical patterns.
3) Connect signals to policy indicators ONLY if supported by the excerpts.
4) Output JSON ONLY with:
   - suspicious_signals: list[str]
   - matched_typologies: list[{{typology: str, rationale: str}}]
   - policy_links: list[{{doc_id: str, section: str, quote: str}}]
   - risk_score: number (0-100)
   - confidence: number (0-1)

Do not output prose. Do not invent policy quotes.
"""

class FraudReasoningAgent:
    def __init__(self):
        self.deepseek = DeepSeekClient()
        self.openai = OpenAIChatClient()

    async def run(self, tx_signals: Dict[str, Any], policy: Dict[str, Any], fraud_patterns: list) -> Dict[str, Any]:
        payload = {
            "transaction_signals": tx_signals,
            "policy_retrieved": policy.get("retrieved", []),
            "fraud_patterns": fraud_patterns,
        }
        prompt = REASONING_PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False)

        raw = None
        try:
            if self.deepseek.available():
                raw = self.deepseek.complete(prompt)
            else:
                raw = await self.openai.complete(prompt, temperature=0.1)
        except Exception:
            logger.exception("Reasoning model failed; falling back to OpenAI")
            raw = await self.openai.complete(prompt, temperature=0.1)

        # best-effort JSON parse
        try:
            return json.loads(raw)
        except Exception:
            # attempt to extract json block
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start:end+1])
            raise
