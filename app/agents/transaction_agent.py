from __future__ import annotations
from typing import Dict, Any, Optional, List
from sqlalchemy import text
from app.db.postgres import AsyncSessionLocal
from app.utils.logger import get_logger

logger = get_logger(__name__)

class TransactionDataAgent:
    """
    Pulls transaction(s) from DB and computes lightweight fraud signals.
    """
    async def fetch_tx(self, tx_id: int) -> Optional[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            res = await session.execute(text("SELECT * FROM transactions WHERE id = :id"), {"id": tx_id})
            row = res.mappings().first()
            return dict(row) if row else None

    async def fetch_recent_for_card(self, card_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            res = await session.execute(
                text("SELECT * FROM transactions WHERE card_id = :c ORDER BY ts DESC LIMIT :n"),
                {"c": card_id, "n": limit},
            )
            return [dict(r) for r in res.mappings().all()]

    def compute_signals(self, tx: Dict[str, Any], recent: List[Dict[str, Any]]) -> Dict[str, Any]:
        amount = float(tx["amount"])
        high_amount = amount >= 5000

        # velocity: count tx in last ~1 hour (synthetic using last N because timestamps are strings in MVP)
        # In production you'd parse ts and filter properly.
        velocity = min(len(recent), 20)
        velocity_anomaly = velocity >= 3

        # geo mismatch: if recent includes another country than current within last few tx
        recent_countries = [r["country"] for r in recent[:5] if r.get("country")]
        geo_mismatch = any(c != tx["country"] for c in recent_countries)

        # merchant heuristics
        merchant = (tx.get("merchant") or "").lower()
        cash_like = any(k in merchant for k in ["crypto", "exchange", "casino", "money", "atm"])

        return {
            "tx_id": tx["id"],
            "card_id": tx["card_id"],
            "amount": amount,
            "currency": tx["currency"],
            "merchant": tx["merchant"],
            "merchant_category": tx["merchant_category"],
            "country": tx["country"],
            "channel": tx["channel"],
            "is_flagged": bool(tx["is_flagged"]),
            "signals": {
                "high_amount": high_amount,
                "velocity_anomaly": velocity_anomaly,
                "geo_mismatch": geo_mismatch,
                "cash_like_or_crypto_merchant": cash_like,
                "velocity_count_proxy": velocity,
            },
            "confidence": 0.75,
        }

    async def run(self, tx_id: int) -> Dict[str, Any]:
        tx = await self.fetch_tx(tx_id)
        if not tx:
            return {"error": f"transaction_not_found:{tx_id}"}
        recent = await self.fetch_recent_for_card(tx["card_id"], limit=20)
        return self.compute_signals(tx, recent)
