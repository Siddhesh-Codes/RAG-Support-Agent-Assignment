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
            api_key=api_key or config.LLM_API_KEY,
            model=model or config.LLM_MODEL,
        )
    elif name == "openai":
        return OpenAIProvider(
            api_key=api_key or config.LLM_API_KEY,
            model=model or config.LLM_MODEL,
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
            api_key=api_key or config.LLM_API_KEY,
            model=model or config.EMBEDDING_MODEL,
        )
    elif name == "local" or name == "offline":
        return LocalEmbeddingProvider()
    elif name == "openai":
        return OpenAIEmbeddingProvider(
            api_key=api_key or config.LLM_API_KEY,
            model=model or config.EMBEDDING_MODEL,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {name}. Supported: 'gemini', 'local', 'openai'")
