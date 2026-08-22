"""Factory functions for instantiating LLM and Embedding providers."""

import os
from src.llm_provider import LLMProvider
from src.rag.index import EmbeddingProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers.gemini_embedding import GeminiEmbeddingProvider
from src.providers.local_embedding import LocalEmbeddingProvider
from src.providers.openai_provider import OpenAIProvider, OpenAIEmbeddingProvider
from src import config


def get_llm_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """Create and return configured LLM provider."""
    name = (provider_name or config.LLM_PROVIDER).lower()

    if name == "gemini":
        return GeminiProvider(
            api_key=api_key or os.getenv("GEMINI_API_KEY") or config.LLM_API_KEY,
            model=model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        )
    elif name == "openai":
        return OpenAIProvider(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or config.LLM_API_KEY,
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o"),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {name}. Supported: 'gemini', 'openai'")


def get_embedding_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> EmbeddingProvider:
    """Create and return configured Embedding provider."""
    name = (provider_name or config.EMBEDDING_PROVIDER).lower()

    if name == "gemini":
        return GeminiEmbeddingProvider(
            api_key=api_key or os.getenv("GEMINI_API_KEY") or config.LLM_API_KEY,
            model=model or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        )
    elif name in ("local", "offline"):
        return LocalEmbeddingProvider()
    elif name == "openai":
        return OpenAIEmbeddingProvider(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or config.LLM_API_KEY,
            model=model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {name}. Supported: 'gemini', 'local', 'openai'")
