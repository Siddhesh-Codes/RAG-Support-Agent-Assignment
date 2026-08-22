"""Naive baseline agent: reproduces the customer's prior failed prototype.

Deliberately anti-pattern design, used only by `python eval/run_eval.py --baseline`
to produce a real, runnable baseline for the README comparison:

- Dumps the ENTIRE knowledge base (including superseded/draft/internal docs)
  into every prompt instead of retrieving relevant passages.
- Dumps the ENTIRE orders.json into the prompt instead of a lookup tool.
- No trust boundaries around retrieved data (injections in KB/docs execute).
- No precedence/conflict handling, no citations, no deterministic abstention.

This makes the four reported failure modes observable:
conflicting policy answers, invented order info, no citations, leaked PII.
"""

import json
import uuid
from pathlib import Path
from typing import Optional

from src import config
from src.llm_provider import LLMProvider
from src.providers.factory import get_llm_provider
from src.response import AgentResponse, ResponseState

BASELINE_SYSTEM_PROMPT = (
    "You are a customer support agent for Aster & Row, an ecommerce store. "
    "Answer the customer's question using the company documents provided. "
    "Be helpful and confident."
)


class _Session:
    def __init__(self):
        self.session_id = str(uuid.uuid4())


class _SessionManagerShim:
    """Minimal stand-in matching the interface run_evaluation expects."""

    def create_session(self) -> _Session:
        return _Session()


class BaselineAgent:
    """Naive RAG-less, tool-less prototype agent for baseline measurement."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.session_manager = _SessionManagerShim()
        self.llm = llm_provider or get_llm_provider()
        kb_dir = Path(config.KNOWLEDGE_BASE_DIR)
        self.full_corpus = "\n\n".join(
            p.read_text(encoding="utf-8") for p in sorted(kb_dir.glob("*.md"))
        )
        with open(config.ORDERS_JSON_PATH, "r", encoding="utf-8") as f:
            self.full_orders = f.read()
        self._history: dict[str, list[dict]] = {}

    def process_message(
        self, user_message: str, session_id: Optional[str] = None
    ) -> AgentResponse:
        session_id = session_id or str(uuid.uuid4())
        history = self._history.setdefault(session_id, [])

        prompt = (
            f"{BASELINE_SYSTEM_PROMPT}\n\n"
            "=== COMPANY DOCUMENTS ===\n"
            f"{self.full_corpus}\n\n"
            "=== ORDER DATABASE ===\n"
            f"{self.full_orders}\n\n"
        )
        messages = [{"role": "system", "content": prompt}]
        for turn in history[-6:]:
            messages.append(turn)
        messages.append({"role": "user", "content": user_message})

        try:
            llm_res = self.llm.chat(messages)
            text = llm_res.text
        except Exception as e:  # baseline has no fallbacks; record the error
            text = f"[baseline error: {e}]"

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": text})

        return AgentResponse(
            state=ResponseState.ANSWER,
            message=text,
            sources=[],
            tool_called=False,
            handoff_recommended=False,
        )
