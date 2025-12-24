from typing import List
from app.models.openai_client import OpenAIEmbedClient

_embedder = None

def get_embedder() -> OpenAIEmbedClient:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbedClient()
    return _embedder

def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedder().embed(texts)
