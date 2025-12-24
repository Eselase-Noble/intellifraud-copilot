import json, time
from pathlib import Path

LOG_PATH = Path("./data/audit.log")

def log(event: dict) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    event = dict(event)
    event["ts"] = time.time()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
