"""Session context management for multi-turn conversations.

Design decisions:
- Each session gets a unique ID (prevents cross-session leakage)
- Tracks: conversation history, last order ID, last topic
- Conversation history is bounded (last N turns) to avoid context bloat
- Sessions are isolated: Session A never sees Session B's state
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional


MAX_HISTORY_TURNS = 10  # Keep last N user+assistant turn pairs


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str       # "user" or "assistant"
    content: str


@dataclass
class SessionContext:
    """Isolated session state for one conversation."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: list[ConversationTurn] = field(default_factory=list)
    last_order_id: Optional[str] = None
    last_topic: Optional[str] = None

    def add_user_message(self, content: str) -> None:
        """Record a user message."""
        self.history.append(ConversationTurn(role="user", content=content))
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """Record an assistant response."""
        self.history.append(ConversationTurn(role="assistant", content=content))
        self._trim_history()

    def set_order_context(self, order_id: str) -> None:
        """Track the most recently referenced order."""
        self.last_order_id = order_id

    def set_topic(self, topic: str) -> None:
        """Track the current conversation topic."""
        self.last_topic = topic

    def get_history_for_llm(self) -> list[dict]:
        """Format conversation history for the LLM context.

        Returns list of {role, content} dicts.
        """
        return [{"role": t.role, "content": t.content} for t in self.history]

    def get_context_summary(self) -> str:
        """Human-readable summary of current session context."""
        parts = [f"Session: {self.session_id[:8]}"]
        if self.last_order_id:
            parts.append(f"Last order: {self.last_order_id}")
        if self.last_topic:
            parts.append(f"Topic: {self.last_topic}")
        parts.append(f"Turns: {len(self.history)}")
        return " | ".join(parts)

    def _trim_history(self) -> None:
        """Keep only the last MAX_HISTORY_TURNS * 2 messages (user+assistant pairs)."""
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]


class SessionManager:
    """Manages isolated sessions. Ensures no cross-session leakage."""

    def __init__(self):
        self._sessions: dict[str, SessionContext] = {}

    def create_session(self) -> SessionContext:
        """Create a new isolated session."""
        session = SessionContext()
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get an existing session by ID. Returns None if not found."""
        return self._sessions.get(session_id)

    def list_session_ids(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())
