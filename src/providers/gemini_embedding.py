"""Gemini embedding provider using google-genai SDK."""

import os
from typing import Optional
from google import genai
from src.rag.index import EmbeddingProvider


class GeminiEmbeddingProvider:
    """Embedding provider using Google Gemini text-embedding-004."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-004"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using Gemini API."""
        if not texts:
            return []

        # Gemini embed_content supports batching or single calls
        # Batch requests in chunks of 50 to avoid any API payload limits
        all_embeddings: list[list[float]] = []
        batch_size = 50

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            max_retries = 8
            base_delay = 2.0
            last_exc = None
            import time
            import re

            for attempt in range(max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                    )
                    for emb in response.embeddings:
                        all_embeddings.append(emb.values)
                    break
                except Exception as e:
                    last_exc = e
                    err = str(e)
                    if "503" in err or "429" in err or "UNAVAILABLE" in err or "RESOURCE_EXHAUSTED" in err:
                        delay = base_delay * (1.8 ** attempt)
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err, re.IGNORECASE)
                        if match:
                            delay = max(delay, float(match.group(1)) + 1.0)
                        elif "429" in err or "RESOURCE_EXHAUSTED" in err:
                            delay = max(delay, 9.0)
                        time.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(f"Gemini embedding API call failed: {str(e)}") from e
            else:
                raise RuntimeError(f"Gemini embedding failed after retries: {str(last_exc)}") from last_exc

        return all_embeddings
