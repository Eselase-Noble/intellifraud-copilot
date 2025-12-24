from __future__ import annotations
import json, re
from typing import Dict, Any, Optional
from app.models.openai_client import OpenAIChatClient
from app.agents.transaction_agent import TransactionDataAgent
from app.agents.policy_rag_agent import PolicyRAGAgent
from app.agents.fraud_reasoning_agent import FraudReasoningAgent
from app.agents.explanation_agent import ExplanationAgent
from app.utils.logger import get_logger
from app.utils.config import settings

logger = get_logger(__name__)

PLANNER_PROMPT = """You are a Planner Agent for a fraud investigation copilot.

Your job:
1) Decide if the query references a specific transaction id.
2) Decide which agents to invoke.
3) Return JSON ONLY with:
   - tx_id: integer | null
   - agents: list[str] from ["TransactionDataAgent","PolicyRAGAgent","FraudReasoningAgent","ExplanationAgent"]
   - policy_question: string (what to search in policy)

Rules:
- Always include TransactionDataAgent if tx_id is present.
- Always include PolicyRAGAgent for policy_question.
- Keep the plan minimal.

User query:
{query}
"""

def _extract_tx_id_fallback(q: str) -> Optional[int]:
    m = re.search(r"(?:tx|transaction)\s*#?\s*(\d{3,})", q, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d{5,})\b", q)
    if m:
        return int(m.group(1))
    return None

class PlannerAgent:
    def __init__(self):
        self.llm = OpenAIChatClient()
        self.tx_agent = TransactionDataAgent()
        self.policy_agent = PolicyRAGAgent()
        self.reasoning_agent = FraudReasoningAgent()
        self.explain_agent = ExplanationAgent()

    async def _make_plan(self, query: str) -> Dict[str, Any]:
        raw = await self.llm.complete(PLANNER_PROMPT.format(query=query), temperature=0.1)
        try:
            plan = json.loads(raw)
        except Exception:
            s = raw.find("{"); e = raw.rfind("}")
            plan = json.loads(raw[s:e+1])

        if plan.get("tx_id") is None:
            plan["tx_id"] = _extract_tx_id_fallback(query)

        if "agents" not in plan or not plan["agents"]:
            plan["agents"] = ["PolicyRAGAgent", "FraudReasoningAgent", "ExplanationAgent"]
            if plan["tx_id"] is not None:
                plan["agents"].insert(0, "TransactionDataAgent")

        if not plan.get("policy_question"):
            plan["policy_question"] = query

        return plan

    async def run(self, query: str) -> Dict[str, Any]:
        plan = await self._make_plan(query)
        logger.info("Plan: %s", plan)

        tx_signals = None
        if plan.get("tx_id") is not None and "TransactionDataAgent" in plan["agents"]:
            tx_signals = await self.tx_agent.run(int(plan["tx_id"]))
        else:
            tx_signals = {"note": "no_tx_id_provided", "signals": {}}

        policy = {"status": "SKIPPED", "retrieved": []}
        if "PolicyRAGAgent" in plan["agents"]:
            policy = self.policy_agent.run(plan["policy_question"])

        # load historical fraud patterns (seeded json)
        import json as _json
        from pathlib import Path
        fraud_path = Path("./data/fraud_cases.json")
        fraud_patterns = _json.loads(fraud_path.read_text(encoding="utf-8")) if fraud_path.exists() else []

        reasoning = {}
        if "FraudReasoningAgent" in plan["agents"]:
            reasoning = await self.reasoning_agent.run(tx_signals, policy, fraud_patterns)

        explained = {}
        if "ExplanationAgent" in plan["agents"]:
            explained = await self.explain_agent.run(reasoning, policy.get("retrieved", []))

        return {
            "plan": plan,
            "transaction": tx_signals,
            "policy": {"status": policy.get("status"), "retrieved_count": len(policy.get("retrieved", []))},
            "reasoning": reasoning,
            "result": explained,
        }
