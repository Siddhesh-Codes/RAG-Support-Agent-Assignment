"""Structured JSONL logging for observability.

Logs every important decision point:
- User message
- Retrieved chunks with scores
- Precedence decisions
- Tool calls with sanitized args/results
- Final response state
- Errors and fallbacks

NEVER logs: API keys, system prompts, internal credentials, raw customer PII.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

from src import config


class TraceLogger:
    """Structured trace logger for a single request/response cycle."""

    def __init__(self, session_id: str, log_dir: Optional[Path] = None):
        self.session_id = session_id
        self.trace_id = f"{int(time.time()*1000)}"
        self.events: list[dict] = []
        self.log_dir = log_dir
        self._enabled = config.DEBUG_TRACE

    def _event(self, event_type: str, data: dict) -> None:
        """Record a trace event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "event": event_type,
            **data,
        }
        self.events.append(entry)

        if self._enabled:
            # Write to stderr for immediate visibility
            print(json.dumps(entry, default=str), file=sys.stderr)

    def log_user_message(self, message: str) -> None:
        self._event("user_message", {"message": message})

    def log_context(self, context_summary: str) -> None:
        self._event("session_context", {"context": context_summary})

    def log_retrieval(self, query: str, results: list[dict]) -> None:
        """Log retrieval results. results should be pre-sanitized."""
        self._event("retrieval", {
            "query": query,
            "result_count": len(results),
            "results": results,
        })

    def log_precedence(self, result: str, explanation: str,
                       authoritative_sources: list[str],
                       excluded_sources: list[str]) -> None:
        self._event("precedence_decision", {
            "result": result,
            "explanation": explanation,
            "authoritative_sources": authoritative_sources,
            "excluded_sources": excluded_sources,
        })

    def log_tool_call(self, tool_name: str, args: dict, result: dict) -> None:
        """Log a tool call. Args and result must already be sanitized."""
        self._event("tool_call", {
            "tool": tool_name,
            "arguments": args,
            "result": result,
        })

    def log_response(self, state: str, message_preview: str,
                     sources: list[str], handoff: bool) -> None:
        self._event("response", {
            "state": state,
            "message_preview": message_preview[:200],
            "sources": sources,
            "handoff": handoff,
        })

    def log_error(self, error_type: str, message: str) -> None:
        self._event("error", {
            "error_type": error_type,
            "message": message,
        })

    def flush_to_file(self) -> None:
        """Write all events to a JSONL log file."""
        if not self.log_dir:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"trace_{self.trace_id}.jsonl"

        with open(log_path, "a", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event, default=str) + "\n")
