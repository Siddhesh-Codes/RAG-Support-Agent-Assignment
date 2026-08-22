"""Tests for the document precedence engine."""

import pytest
from src.rag.ingest import Chunk, DocumentMetadata
from src.rag.precedence import (
    classify_chunk,
    analyze_precedence,
    filter_by_supersession,
    ScoredChunk,
    PrecedenceResult,
)


def _make_chunk(source_file: str, heading: str = "Section",
                status: str = "active", authority: str = "official",
                audience: str = "customer", doc_id: str = "",
                supersedes: str = "", superseded_by: str = "",
                customer_answering=None) -> Chunk:
    """Helper to create test chunks with specific metadata."""
    return Chunk(
        text="Test content",
        source_file=source_file,
        heading=heading,
        metadata=DocumentMetadata(
            document_id=doc_id,
            status=status,
            policy_authority=authority,
            audience=audience,
            supersedes=supersedes,
            superseded_by=superseded_by,
            customer_answering=customer_answering,
        ),
    )


def _scored(chunk: Chunk, score: float = 0.85) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score)


# --- Classification ---

class TestClassifyChunk:
    def test_active_official_customer_is_authoritative(self):
        chunk = _make_chunk("01-returns.md", status="active",
                            authority="official", audience="customer")
        assert classify_chunk(chunk) == "authoritative"

    def test_superseded_is_superseded(self):
        chunk = _make_chunk("02-returns-legacy.md", status="superseded",
                            authority="official")
        assert classify_chunk(chunk) == "superseded"

    def test_draft_is_excluded(self):
        chunk = _make_chunk("14-migration.md", status="draft",
                            authority="none")
        assert classify_chunk(chunk) == "excluded"

    def test_internal_official_is_internal_guidance(self):
        chunk = _make_chunk("13-escalation.md", status="active",
                            authority="official", audience="internal")
        assert classify_chunk(chunk) == "internal_guidance"

    def test_no_authority_is_excluded(self):
        chunk = _make_chunk("14-migration.md", status="active",
                            authority="none")
        assert classify_chunk(chunk) == "excluded"

    def test_customer_answering_false_is_excluded(self):
        chunk = _make_chunk("14-migration.md", status="draft",
                            authority="none", customer_answering=False)
        assert classify_chunk(chunk) == "excluded"


# --- Precedence Analysis ---

class TestAnalyzePrecedence:
    def test_empty_results_is_insufficient(self):
        decision = analyze_precedence([])
        assert decision.result == PrecedenceResult.INSUFFICIENT

    def test_single_authoritative_source(self):
        chunk = _make_chunk("01-returns.md", doc_id="RET-2026-01")
        decision = analyze_precedence([_scored(chunk)])
        assert decision.result == PrecedenceResult.AUTHORITATIVE
        assert len(decision.authoritative_chunks) == 1

    def test_superseded_only_returns_superseded_result(self):
        chunk = _make_chunk("02-legacy.md", status="superseded", doc_id="RET-2024-01")
        decision = analyze_precedence([_scored(chunk)])
        assert decision.result == PrecedenceResult.SUPERSEDED_ONLY

    def test_excluded_only_returns_insufficient(self):
        chunk = _make_chunk("14-migration.md", status="draft", authority="none")
        decision = analyze_precedence([_scored(chunk)])
        assert decision.result == PrecedenceResult.INSUFFICIENT

    def test_current_beats_legacy_when_supersession_exists(self):
        """When current and legacy returns policy are both retrieved,
        and there's a supersession relationship, this is NOT a conflict."""
        current = _make_chunk("01-returns.md", doc_id="RET-2026-01",
                              supersedes="RET-2024-01")
        legacy = _make_chunk("02-legacy.md", doc_id="RET-2024-01",
                             status="superseded", superseded_by="RET-2026-01")

        # Legacy gets classified as superseded, so only current is authoritative
        decision = analyze_precedence([_scored(current), _scored(legacy)])
        assert decision.result == PrecedenceResult.AUTHORITATIVE
        assert len(decision.authoritative_chunks) == 1
        assert decision.authoritative_chunks[0].chunk.source_file == "01-returns.md"

    def test_genuine_conflict_between_active_sources(self):
        """Two active official sources with no supersession = CONFLICT.
        This is the Breeze Tumbler cleaning case."""
        care = _make_chunk("11-product-care.md", doc_id="CARE-2026-01",
                           heading="Breeze Tumbler")
        card = _make_chunk("12-breeze-tumbler.md", doc_id="PROD-BREEZE-20",
                           heading="Cleaning")

        decision = analyze_precedence([_scored(care, 0.9), _scored(card, 0.88)])
        assert decision.result == PrecedenceResult.CONFLICT
        assert len(decision.conflicting_chunks) == 2

    def test_multiple_sources_same_doc_is_not_conflict(self):
        """Multiple chunks from the same document = not a conflict."""
        c1 = _make_chunk("01-returns.md", heading="Window", doc_id="RET-2026-01")
        c2 = _make_chunk("01-returns.md", heading="Shipping", doc_id="RET-2026-01")
        decision = analyze_precedence([_scored(c1), _scored(c2)])
        assert decision.result == PrecedenceResult.AUTHORITATIVE

    def test_supplementary_docs_not_conflict(self):
        """Returns + TrailPlus are distinct documents with complementary scopes.
        They should both be authoritative, not flagged as a conflict."""
        returns = _make_chunk("01-returns.md", doc_id="RET-2026-01")
        trailplus = _make_chunk("09-trailplus.md", doc_id="MEM-2026-01")

        decision = analyze_precedence([_scored(returns), _scored(trailplus)])
        assert decision.result == PrecedenceResult.AUTHORITATIVE
        assert len(decision.authoritative_chunks) == 2

    def test_draft_excluded_even_with_high_score(self):
        """Draft doc should be excluded even with higher similarity score."""
        draft = _make_chunk("14-migration.md", status="draft",
                            authority="none", doc_id="MIG-TEST-04")
        active = _make_chunk("01-returns.md", doc_id="RET-2026-01")

        decision = analyze_precedence([
            _scored(draft, score=0.95),  # Higher score!
            _scored(active, score=0.80),
        ])
        assert decision.result == PrecedenceResult.AUTHORITATIVE
        # Draft should be in excluded, not authoritative
        auth_files = [sc.chunk.source_file for sc in decision.authoritative_chunks]
        assert "14-migration.md" not in auth_files
        assert "01-returns.md" in auth_files


# --- Supersession Filtering ---

class TestFilterBySupersession:
    def test_removes_superseded_when_current_present(self):
        current = _scored(_make_chunk("01-returns.md", doc_id="RET-2026-01",
                                      supersedes="RET-2024-01"))
        legacy = _scored(_make_chunk("02-legacy.md", doc_id="RET-2024-01",
                                     status="superseded",
                                     superseded_by="RET-2026-01"))

        filtered = filter_by_supersession([current, legacy])
        files = [sc.chunk.source_file for sc in filtered]
        assert "01-returns.md" in files
        assert "02-legacy.md" not in files

    def test_keeps_superseded_when_current_not_present(self):
        legacy = _scored(_make_chunk("02-legacy.md", doc_id="RET-2024-01",
                                     status="superseded",
                                     superseded_by="RET-2026-01"))
        filtered = filter_by_supersession([legacy])
        assert len(filtered) == 1  # Kept because superseding doc isn't in results

    def test_no_supersession_keeps_all(self):
        a = _scored(_make_chunk("05-domestic.md", doc_id="SHIP-US"))
        b = _scored(_make_chunk("06-international.md", doc_id="SHIP-INTL"))
        filtered = filter_by_supersession([a, b])
        assert len(filtered) == 2


# --- Conflict relevance gating ---

def _tumbler_conflict_chunks(warranty_chunk=None):
    """Build the canonical 11 vs 12 Breeze Tumbler conflict retrieval set."""
    care = _scored(_make_chunk(
        "11-product-care.md", heading="Drinkware", doc_id="CARE-01"), score=0.80)
    care.chunk.text = "The Breeze Tumbler stainless-steel body must be hand-washed; only the lid is dishwasher safe."
    card = _scored(_make_chunk(
        "12-breeze-tumbler-product-card.md", heading="Cleaning", doc_id="PROD-12"), score=0.75)
    card.chunk.text = "All components of the Breeze Tumbler are dishwasher safe; top rack recommended."
    chunks = [care, card]
    if warranty_chunk is not None:
        chunks.append(_scored(warranty_chunk, score=0.70))
    return chunks


class TestConflictRelevanceGate:
    """Regression: an unrelated query must not be hijacked by a detected conflict.

    A warranty question about 'drinkware' previously triggered the Breeze
    Tumbler dishwashing conflict because both product-care docs were retrieved
    alongside the warranty doc; the conflict branch then replaced the answer.
    """

    def test_conflict_surfaces_for_relevant_query(self):
        decision = analyze_precedence(
            _tumbler_conflict_chunks(), query="Can I put the Breeze Tumbler in the dishwasher?")
        assert decision.result == PrecedenceResult.CONFLICT

    def test_unrelated_query_bypasses_conflict(self):
        warranty = _make_chunk("07-warranty.md", heading="Coverage", doc_id="WAR-01")
        decision = analyze_precedence(
            _tumbler_conflict_chunks(warranty),
            query="Is the coverage on your drinkware good forever, like a lifetime thing?")
        assert decision.result == PrecedenceResult.AUTHORITATIVE
        auth_files = [sc.chunk.source_file for sc in decision.authoritative_chunks]
        assert auth_files == ["07-warranty.md"]
        # The conflicting chunks must not appear as authority
        assert "11-product-care.md" not in auth_files
        assert "12-breeze-tumbler-product-card.md" not in auth_files

    def test_unrelated_query_with_no_other_sources_is_insufficient(self):
        decision = analyze_precedence(
            _tumbler_conflict_chunks(),
            query="Do your zippers and hardware contain nickel?")
        assert decision.result == PrecedenceResult.INSUFFICIENT

    def test_no_query_still_surfaces_conflict(self):
        # Backwards compatibility: without a query, conflicts always surface.
        decision = analyze_precedence(_tumbler_conflict_chunks())
        assert decision.result == PrecedenceResult.CONFLICT

    def test_nonconflict_sections_of_conflicting_file_are_kept(self):
        """Regression: an irrelevant conflict must not discard relevant
        non-conflict sections (e.g. packing-cube care) from the same file."""
        care_packing = _make_chunk("11-product-care.md", heading="Travel accessories", doc_id="CARE-01")
        care_packing.text = "Packing cubes: machine wash cold, cool water, air dry only. Do not tumble dry on high heat."
        card = _make_chunk("12-breeze-tumbler-product-card.md", heading="Cleaning", doc_id="PROD-12")
        card.text = "All components of the Breeze Tumbler are dishwasher safe."
        decision = analyze_precedence(
            [_scored(care_packing, 0.8), _scored(card, 0.6)],
            query="Can I put my packing cubes in the dryer on high heat?")
        assert decision.result == PrecedenceResult.AUTHORITATIVE
        headings = [sc.chunk.heading for sc in decision.authoritative_chunks]
        assert "Travel accessories" in headings
