import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    sqlite_path: str = os.getenv("SQLITE_PATH", "./data/transactions.sqlite3")
    policy_index_dir: str = os.getenv("POLICY_INDEX_DIR", "./data/policy_index")
    policy_data_dir: str = os.getenv("POLICY_DATA_DIR", "./data/policies")
    top_k: int = int(os.getenv("TOP_K", "5"))
    rerank_k: int = int(os.getenv("RERANK_K", "3"))

settings = Settings()
