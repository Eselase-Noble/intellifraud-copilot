import os
import httpx
from typing import Optional

class DeepSeekClient:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.url = "https://api.deepseek.com/chat/completions"
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, prompt: str, temperature: float = 0.1) -> str:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=60) as client:
            r = client.post(self.url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
