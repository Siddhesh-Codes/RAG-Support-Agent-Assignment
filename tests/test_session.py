"""Tests for session context management and isolation."""

import pytest
from src.session import SessionContext, SessionManager


class TestSessionContext:
    def test_new_session_has_unique_id(self):
        s1 = SessionContext()
        s2 = SessionContext()
        assert s1.session_id != s2.session_id

    def test_add_messages_and_retrieve(self):
        session = SessionContext()
        session.add_user_message("Hello")
        session.add_assistant_message("Hi there!")

        history = session.get_history_for_llm()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there!"}

    def test_order_context_tracking(self):
        session = SessionContext()
        assert session.last_order_id is None
        session.set_order_context("ORD-1007")
        assert session.last_order_id == "ORD-1007"

    def test_topic_tracking(self):
        session = SessionContext()
        assert session.last_topic is None
        session.set_topic("international shipping")
        assert session.last_topic == "international shipping"

    def test_history_trimming(self):
        session = SessionContext()
        # Add more messages than the limit
        for i in range(30):
            session.add_user_message(f"User message {i}")
            session.add_assistant_message(f"Assistant message {i}")

        history = session.get_history_for_llm()
        # Should be trimmed to MAX_HISTORY_TURNS * 2 (default: 20)
        assert len(history) <= 20

    def test_context_summary(self):
        session = SessionContext()
        session.set_order_context("ORD-1007")
        session.set_topic("returns")
        summary = session.get_context_summary()
        assert "ORD-1007" in summary
        assert "returns" in summary


class TestSessionManager:
    def test_create_session(self):
        manager = SessionManager()
        session = manager.create_session()
        assert session.session_id is not None

    def test_get_session(self):
        manager = SessionManager()
        session = manager.create_session()
        retrieved = manager.get_session(session.session_id)
        assert retrieved is session

    def test_get_nonexistent_session(self):
        manager = SessionManager()
        assert manager.get_session("fake-id") is None

    def test_session_isolation(self):
        """CRITICAL: sessions must not share state."""
        manager = SessionManager()
        session_a = manager.create_session()
        session_b = manager.create_session()

        # Set state on session A
        session_a.set_order_context("ORD-1007")
        session_a.add_user_message("Where is ORD-1007?")
        session_a.set_topic("order tracking")

        # Set different state on session B
        session_b.set_order_context("ORD-1012")
        session_b.add_user_message("Check ORD-1012")
        session_b.set_topic("order status")

        # Verify isolation
        assert session_a.last_order_id == "ORD-1007"
        assert session_b.last_order_id == "ORD-1012"

        assert session_a.last_topic == "order tracking"
        assert session_b.last_topic == "order status"

        history_a = session_a.get_history_for_llm()
        history_b = session_b.get_history_for_llm()

        assert history_a[0]["content"] == "Where is ORD-1007?"
        assert history_b[0]["content"] == "Check ORD-1012"

        # Session A must NOT contain session B content
        a_contents = str(history_a)
        assert "ORD-1012" not in a_contents

        # Session B must NOT contain session A content
        b_contents = str(history_b)
        assert "ORD-1007" not in b_contents

    def test_list_sessions(self):
        manager = SessionManager()
        s1 = manager.create_session()
        s2 = manager.create_session()
        ids = manager.list_session_ids()
        assert s1.session_id in ids
        assert s2.session_id in ids
