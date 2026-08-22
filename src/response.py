"""Response state model for the support agent.

Every agent response has an explicit internal state that drives behavior
and makes testing deterministic. The user sees a natural language response;
the internal state drives assertions.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Optional


class ResponseState(Enum):
    """Internal state classification for every agent response."""
    ANSWER = "answer"                          # Confident, grounded answer
    CLARIFICATION_REQUIRED = "clarification"   # Need more info from user
    INSUFFICIENT_INFORMATION = "insufficient"  # KB doesn't cover this
    CONFLICT = "conflict"                      # Active sources disagree
    HANDOFF = "handoff"                        # Recommend human assistance
    REFUSAL = "refusal"                        # Refuse unsafe/injection request


@dataclass
class SourceCitation:
    """A reference to a specific knowledge base source."""
    filename: str
    heading: Optional[str] = None
    document_id: Optional[str] = None

    def format(self) -> str:
        parts = [self.filename]
        if self.heading:
            parts.append(f"- {self.heading}")
        return " ".join(parts)


def strip_markdown(text: str) -> str:
    """Strip markdown bolding, italics, backticks, and header hashes for clean CLI terminal display."""
    if not text:
        return ""
    # Strip bold asterisks (**text** -> text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    # Strip italic asterisks (*text* -> text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    # Strip bold/italic underscores (__text__ or _text_)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"\1", text)
    # Strip backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Strip markdown header markers (e.g. ## Heading -> Heading)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text


@dataclass
class AgentResponse:
    """Structured agent response with metadata for testing."""
    state: ResponseState
    message: str
    sources: list[SourceCitation] = field(default_factory=list)
    tool_called: bool = False
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[dict] = None
    handoff_recommended: bool = False
    order_id: Optional[str] = None

    def has_source(self, filename: str) -> bool:
        """Check if a specific source file was cited."""
        return any(s.filename == filename for s in self.sources)

    def source_filenames(self) -> list[str]:
        """Get list of cited source filenames."""
        return [s.filename for s in self.sources]

    def format_for_user(self) -> str:
        """Format the response for the user-facing CLI with clean plain-text formatting."""
        clean_msg = strip_markdown(self.message.strip())
        parts = [clean_msg]

        # Only append structured sources block if not already formatted inline in message
        if self.sources and not any(header in clean_msg for header in ["Sources:", "Source:"]):
            parts.append("\nSources:")
            for src in self.sources:
                parts.append(f"  - {src.format()}")

        if self.handoff_recommended and not any(phrase in clean_msg.lower() for phrase in ["contact support", "human support specialist", "human assistance"]):
            parts.append("\nI recommend contacting a human support specialist for further assistance.")

        return "\n".join(parts)
