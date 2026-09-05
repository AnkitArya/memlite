"""Embedding abstraction for MemLite.

Mirrors mem0's `mem0.embeddings.openai.OpenAIEmbedding`: an OpenAI-compatible
embeddings client. Point it at DeepInfra, NVIDIA, OpenAI, Groq, etc. by setting
`openai_base_url` (defaults to OpenAI, key from OPENAI_API_KEY).
"""
import os

import openai


class Embedder:
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.model = config.get("model", "BAAI/bge-base-en-v1.5")
        self.base_url = config.get(
            "openai_base_url", "https://api.deepinfra.com/v1/openai"
        )
        # Prefer an inline key, then OPENAI_API_KEY, then DEEPINFRA_API_KEY
        api_key = (
            config.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEEPINFRA_API_KEY")
        )
        self.dims = config.get("embedding_dims")  # optional, verified at runtime
        self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url)

    def embed(self, text: str) -> list[float]:
        r = self.client.embeddings.create(model=self.model, input=text)
        return r.data[0].embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        r = self.client.embeddings.create(model=self.model, input=texts)
        # Preserve order
        return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
