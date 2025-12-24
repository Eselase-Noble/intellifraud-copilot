import os
from typing import List
from openai import AsyncOpenAI, OpenAI
from app.utils.config import settings

class OpenAIChatClient:
    def __init__(self):
        # print("API KEY", os.getenv("OPENAI_API_KEY"))
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def complete(self, prompt: str, temperature: float = 0.2) -> str:
        resp = await self.client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

class OpenAIEmbedClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=settings.openai_embed_model, input=texts)
        return [d.embedding for d in resp.data]
