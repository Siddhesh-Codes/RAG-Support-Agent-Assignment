"""LLM provider abstraction layer.

Clean interface so core application is independent of Gemini/OpenAI.
Only the provider implementation knows about vendor-specific APIs.

Architecture:
  Agent → LLMProvider interface → GeminiProvider (or OpenAIProvider)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolCall:
    """Represents a tool/function call requested by the model."""
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str                              # The model's text response
    tool_calls: list[ToolCall]             # Any tool calls the model wants to make
    finish_reason: Optional[str] = None    # "stop", "tool_calls", etc.

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """Abstract interface for LLM providers.

    Implementations handle vendor-specific API details.
    The agent only interacts through this interface.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_results: Optional[list[dict]] = None,
    ) -> LLMResponse:
        """Send a chat request to the model.

        Args:
            messages: List of {role, content} dicts
            tools: Optional tool/function schemas
            tool_results: Optional results from previous tool calls

        Returns:
            LLMResponse with text and/or tool calls
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier being used."""
        ...
