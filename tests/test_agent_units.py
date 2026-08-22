"""Unit tests for AgentOrchestrator helper methods (no LLM required)."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent import AgentOrchestrator


def _make_agent() -> AgentOrchestrator:
    # Helper methods under test do not touch the LLM, retriever, or tools.
    return AgentOrchestrator(llm_provider=None, retriever=None, order_tool=None)


class TestTransactionalRequestDetection:
    """Regression: _is_transactional_request must return a bool.

    It previously built its pattern list but fell through without a return
    statement, so it always evaluated to None (falsy) and unsupported
    transactional requests never triggered the deterministic handoff path.
    """

    def test_returns_true_for_cancel_order(self):
        assert _make_agent()._is_transactional_request("please cancel my order") is True

    def test_returns_true_for_refund_and_address_change(self):
        agent = _make_agent()
        assert agent._is_transactional_request("Can you process a refund for me?") is True
        assert agent._is_transactional_request("I need to change my shipping address") is True

    def test_returns_false_for_plain_questions(self):
        agent = _make_agent()
        assert agent._is_transactional_request("What is your return window?") is False
        assert agent._is_transactional_request("Where is ORD-1007?") is False

    def test_returns_bool_not_none_for_any_input(self):
        # The original bug: every call returned None.
        result = _make_agent()._is_transactional_request("cancel the order please")
        assert isinstance(result, bool)
        assert result is True


class TestOrderInquiryDetection:
    def test_paraphrased_order_inquiries_detected(self):
        agent = _make_agent()
        assert agent._is_order_inquiry("Where's my package?") is True
        assert agent._is_order_inquiry("Has my stuff shipped yet?") is True
        assert agent._is_order_inquiry("What is the return policy?") is False
