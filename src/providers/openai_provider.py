"""OpenAI LLM and Embedding provider implementation."""

import json
import os
import sys
import time
from typing import Optional
from src.llm_provider import LLMProvider, LLMResponse, ToolCall
from src.rag.index import EmbeddingProvider


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using openai Python SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is not set. Please provide it in .env or via environment.")
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install openai package to use OpenAIProvider: pip install openai")
        return self._client

    def get_model_name(self) -> str:
        return self.model

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_results: Optional[list[dict]] = None,
    ) -> LLMResponse:
        """Execute chat completion with tool calling support."""
        formatted_messages = list(messages)
        if tool_results:
            for tr in tool_results:
                formatted_messages.append({
                    "role": "tool",
                    "name": tr.get("tool_name"),
                    "content": str(tr.get("result")),
                })

        kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]

        max_retries = 6
        base_delay = 2.0
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                last_exception = e
                err_msg = str(e)
                if "429" in err_msg or "rate_limit" in err_msg.lower() or "503" in err_msg or "overloaded" in err_msg.lower():
                    delay = base_delay * (1.8 ** attempt)
                    print(f"[{self.model}] Rate limit / server busy (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s...", file=sys.stderr, flush=True)
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError(f"OpenAI API call failed: {str(e)}") from e
        else:
            raise RuntimeError(f"OpenAI API call failed after {max_retries} retries: {str(last_exception)}") from last_exception

        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_calls.append(ToolCall(name=tc.function.name, arguments=args))

        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
        )


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI Embedding provider using text-embedding-3-small."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install openai package: pip install openai")
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        max_retries = 6
        base_delay = 2.0
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                last_exception = e
                err_msg = str(e)
                if "429" in err_msg or "rate_limit" in err_msg.lower() or "503" in err_msg:
                    delay = base_delay * (1.8 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError(f"OpenAI embedding call failed: {str(e)}") from e
        raise RuntimeError(f"OpenAI embedding call failed after retries: {str(last_exception)}") from last_exception

