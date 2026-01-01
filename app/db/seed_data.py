from __future__ import annotations
import csv, json, os
from pathlib import Path
import asyncio
from sqlalchemy import text
from app.db.postgres import engine
from app.utils.config import settings
from app.utils.logger import get_logger
from app.rag.retriever import PolicyIndex


#author: Noble Eselase Vulley
#version: 1.0.0


logger = get_logger(__name__)

DATA_DIR = Path("./data")
TX_CSV = DATA_DIR / "transactions.csv"
FRAUD_JSON = DATA_DIR / "fraud_cases.json"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  card_id TEXT NOT NULL,
  amount REAL NOT NULL,
  currency TEXT NOT NULL,
  merchant TEXT NOT NULL,
  merchant_category TEXT NOT NULL,
  country TEXT NOT NULL,
  city TEXT NOT NULL,
  lat REAL,
  lon REAL,
  channel TEXT NOT NULL,
  is_flagged INTEGER NOT NULL
);
"""

async def init_db():
    db_path = Path(settings.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.execute(text(SCHEMA_SQL))
        await conn.commit()

async def load_transactions():
    if not TX_CSV.exists():
        raise RuntimeError(f"Missing {TX_CSV}. Run from repo root.")
    async with engine.begin() as conn:
        # quick reset for MVP
        await conn.execute(text("DELETE FROM transactions;"))
        with open(TX_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                await conn.execute(
                    text(" INSERT INTO transactions (id, ts, card_id, amount, currency, merchant, merchant_category, country, city, lat, lon, channel, is_flagged) VALUES (:id, :ts, :card_id, :amount, :currency, :merchant, :merchant_category, :country, :city, :lat, :lon, :channel, :is_flagged)"),
                    {
                        "id": int(row["id"]),
                        "ts": row["ts"],
                        "card_id": row["card_id"],
                        "amount": float(row["amount"]),
                        "currency": row["currency"],
                        "merchant": row["merchant"],
                        "merchant_category": row["merchant_category"],
                        "country": row["country"],
                        "city": row["city"],
                        "lat": float(row["lat"]) if row["lat"] else None,
                        "lon": float(row["lon"]) if row["lon"] else None,
                        "channel": row["channel"],
                        "is_flagged": int(row["is_flagged"]),
                    }
                )
        await conn.commit()

def ensure_sample_files():
    DATA_DIR.mkdir(exist_ok=True)
    # transactions.csv
    if not TX_CSV.exists():
        with open(TX_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id","ts","card_id","amount","currency","merchant","merchant_category","country","city","lat","lon","channel","is_flagged"])
            w.writerow([89342,"2025-12-20T14:03:12Z","card_001",9800,"EUR","CryptoExchangeX","financial_services","DE","Berlin",52.52,13.405,"card_present",1])
            w.writerow([89343,"2025-12-20T14:15:09Z","card_001",120,"EUR","CoffeeSpot","food_beverage","DE","Berlin",52.5201,13.4049,"card_present",0])
            w.writerow([89344,"2025-12-20T14:18:21Z","card_001",5000,"EUR","ElectroMart","electronics","FR","Paris",48.8566,2.3522,"ecom",1])
            w.writerow([89345,"2025-12-20T15:01:55Z","card_002",40,"USD","RideShare","transport","US","New York",40.7128,-74.0060,"in_app",0])

    if not FRAUD_JSON.exists():
        cases = [
            {"case_id":"f_1001","typology":"Account takeover","signals":["new_device","ecom_spike","password_reset"],"notes":"Rapid e-com purchases after credential compromise"},
            {"case_id":"f_1002","typology":"Bust-out fraud","signals":["credit_line_ramp","high_amount","cash_like_merchant"],"notes":"Increasing spend culminating in large cash-like transaction"},
            {"case_id":"f_1003","typology":"Card testing","signals":["many_small_tx","declines","multiple_merchants"],"notes":"High volume low-value authorizations"}
        ]
        FRAUD_JSON.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    # policies
    pol_dir = Path(settings.policy_data_dir)
    pol_dir.mkdir(parents=True, exist_ok=True)
    aml = pol_dir / "aml_policy.txt"
    if not aml.exists():
        aml.write_text(
            """# AML / Fraud Monitoring Policy (Sample)

## High-risk transaction indicators
- High-amount transactions above local thresholds (e.g., > 5000 EUR) require enhanced review.
- Cross-border activity inconsistent with customer history is considered elevated risk.
- Rapid successive authorizations (velocity) in a short time window can indicate card testing or account compromise.
- Transactions involving **cash-like** or **crypto exchange** merchants require additional scrutiny and may trigger alerts.

## Common fraud typologies
- Card testing: many small authorizations across multiple merchants, often with declines.
- Account takeover: anomalous location/device + sudden e-commerce spike.
- Bust-out: ramping credit utilization followed by large purchases or cash-like merchant activity.

## Analyst guidance
- Prefer corroborating evidence (merchant category, location anomalies, historical patterns) before escalation.
- Document decision rationale with references to the policy indicators above.
""", encoding="utf-8"
        )

async def main():
    ensure_sample_files()
    await init_db()
    await load_transactions()
    # build policy index
    idx = PolicyIndex()
    idx.build_or_load(force=True)
    logger.info("Seed complete. DB=%s, policy index=%s", settings.sqlite_path, settings.policy_index_dir)

if __name__ == "__main__":
    asyncio.run(main())
