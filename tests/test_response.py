"""Tests for the response state model."""

import pytest
from src.response import ResponseState, SourceCitation, AgentResponse


class TestResponseState:
    def test_all_states_defined(self):
        expected = {"answer", "clarification", "insufficient", "conflict", "handoff", "refusal"}
        actual = {s.value for s in ResponseState}
        assert actual == expected


class TestSourceCitation:
    def test_format_with_heading(self):
        citation = SourceCitation(filename="01-returns.md", heading="Standard return window")
        assert citation.format() == "01-returns.md - Standard return window"

    def test_format_without_heading(self):
        citation = SourceCitation(filename="01-returns.md")
        assert citation.format() == "01-returns.md"


class TestAgentResponse:
    def test_has_source(self):
        resp = AgentResponse(
            state=ResponseState.ANSWER,
            message="Test",
            sources=[
                SourceCitation(filename="01-returns.md", heading="Window"),
                SourceCitation(filename="09-trailplus.md"),
            ],
        )
        assert resp.has_source("01-returns.md")
        assert resp.has_source("09-trailplus.md")
        assert not resp.has_source("02-legacy.md")

    def test_source_filenames(self):
        resp = AgentResponse(
            state=ResponseState.ANSWER,
            message="Test",
            sources=[SourceCitation(filename="a.md"), SourceCitation(filename="b.md")],
        )
        assert resp.source_filenames() == ["a.md", "b.md"]

    def test_format_with_sources_and_handoff(self):
        resp = AgentResponse(
            state=ResponseState.HANDOFF,
            message="I can't help with that.",
            sources=[SourceCitation(filename="13-escalation.md", heading="Handoff")],
            handoff_recommended=True,
        )
        formatted = resp.format_for_user()
        assert "I can't help with that." in formatted
        assert "13-escalation.md" in formatted
        assert "human support" in formatted.lower()

    def test_tool_metadata(self):
        resp = AgentResponse(
            state=ResponseState.ANSWER,
            message="Your order is shipped.",
            tool_called=True,
            tool_name="order_lookup",
            tool_args={"order_id": "ORD-1007"},
            tool_result={"status": "shipped"},
            order_id="ORD-1007",
        )
        assert resp.tool_called
        assert resp.tool_name == "order_lookup"
        assert resp.order_id == "ORD-1007"
