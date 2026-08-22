"""Document precedence engine.

This is the most critical business-logic component.
It determines which retrieved chunks are authoritative and detects conflicts.

NO LLM dependency. Fully deterministic and testable.

Precedence rules (from assignment + audit):
1. status=superseded documents lose to status=active documents on the same topic
2. status=draft / policy_authority=none documents are NEVER authoritative
3. audience=internal documents provide agent behavior rules, not customer-facing policy
4. Two active/official/customer-facing documents that disagree = GENUINE CONFLICT
5. Similarity score alone never overrides metadata-based precedence
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.rag.ingest import Chunk


# Subject terms for known conflict pairs. A detected conflict only hijacks the
# answer when the user's query actually concerns the conflict's subject. This
# prevents e.g. a warranty question about "drinkware" from being answered as a
# Breeze Tumbler dishwashing conflict just because both product-care docs were
# retrieved alongside the warranty doc.
#
# Two term sets per pair:
# - query hints (broad): decide whether the user's question is about the
#   conflict subject at all;
# - entity terms (narrow): decide which retrieved chunks are part of the
#   conflict when the query is unrelated. Entity terms must be specific enough
#   not to match unrelated care sections (e.g. "hand-washed in cool water" for
#   packing cubes must NOT be treated as part of the tumbler conflict).
_CONFLICT_QUERY_HINTS: dict[frozenset, tuple[str, ...]] = {
    frozenset({"11-product-care.md", "12-breeze-tumbler-product-card.md"}): (
        "breeze", "tumbler", "dishwasher", "dish washing", "wash", "hand-wash",
        "handwash", "clean", "cleaning", "care", "lid",
    ),
}

_CONFLICT_ENTITY_TERMS: dict[frozenset, tuple[str, ...]] = {
    frozenset({"11-product-care.md", "12-breeze-tumbler-product-card.md"}): (
        "breeze", "tumbler", "dishwasher",
    ),
}


def _conflict_relevant_to_query(conflict_files: set[str], query: Optional[str]) -> bool:
    """Is this detected conflict about what the user is actually asking?

    Unknown conflict pairs are conservatively treated as relevant (surface them).
    """
    if query is None:
        return True
    hints = _CONFLICT_QUERY_HINTS.get(frozenset(conflict_files))
    if hints is None:
        return True
    q = query.lower()
    return any(h in q for h in hints)


class PrecedenceResult(Enum):
    """Outcome of precedence analysis."""
    AUTHORITATIVE = "authoritative"         # Clear winner(s)
    CONFLICT = "conflict"                   # Active sources disagree
    INSUFFICIENT = "insufficient"           # No relevant authoritative sources
    SUPERSEDED_ONLY = "superseded_only"     # Only legacy sources found


@dataclass
class ScoredChunk:
    """A chunk with its retrieval similarity score."""
    chunk: Chunk
    score: float  # cosine similarity, higher = more similar


@dataclass
class PrecedenceDecision:
    """Result of precedence analysis on retrieved candidates."""
    result: PrecedenceResult
    authoritative_chunks: list[ScoredChunk]   # Winning chunks
    conflicting_chunks: list[ScoredChunk]     # Chunks that conflict (when result=CONFLICT)
    excluded_chunks: list[ScoredChunk]        # Chunks excluded by precedence
    explanation: str                           # Human-readable rationale


def classify_chunk(chunk: Chunk) -> str:
    """Classify a chunk into a precedence tier.

    Returns one of:
    - "authoritative": active, official — can serve as a source for answers
    - "internal_guidance": active, official, internal — for agent behavior only
    - "superseded": replaced by a newer document
    - "excluded": draft, no authority, or explicitly not for customer answers
    """
    meta = chunk.metadata

    # Draft or no-authority documents are never authoritative
    if meta.is_draft or meta.has_no_authority:
        return "excluded"

    # Explicit customer_answering=false flag
    if meta.customer_answering is False:
        return "excluded"

    # Superseded documents
    if meta.is_superseded:
        return "superseded"

    # Active official documents
    if meta.is_authoritative:
        if meta.is_internal:
            return "internal_guidance"
        return "authoritative"

    # Fallback: not clearly authoritative
    return "excluded"


def _detect_topic_overlap(chunks_a: list[ScoredChunk], chunks_b: list[ScoredChunk]) -> bool:
    """Check if two groups of chunks cover overlapping topics.

    Simple heuristic: if both groups have chunks with high similarity
    to the same query, they likely cover overlapping topics.
    This is already implied by the fact that retrieval returned them
    for the same query.
    """
    # If both exist and were retrieved for the same query, they overlap
    return bool(chunks_a and chunks_b)


def analyze_precedence(scored_chunks: list[ScoredChunk], query: Optional[str] = None) -> PrecedenceDecision:
    """Analyze retrieved chunks and determine which are authoritative.

    This is the core precedence logic:
    1. Classify each chunk by its metadata
    2. If authoritative chunks exist, check for conflicts among them
    3. Handle superseded/excluded chunks appropriately

    When `query` is provided, a detected conflict only takes over the answer if
    the query concerns the conflict's subject; otherwise the conflicting chunks
    are set aside and the remaining authoritative chunks answer the question.
    """
    if not scored_chunks:
        return PrecedenceDecision(
            result=PrecedenceResult.INSUFFICIENT,
            authoritative_chunks=[],
            conflicting_chunks=[],
            excluded_chunks=[],
            explanation="No relevant content found in the knowledge base.",
        )

    # Classify all chunks
    authoritative = []
    internal = []
    superseded = []
    excluded = []

    for sc in scored_chunks:
        tier = classify_chunk(sc.chunk)
        if tier == "authoritative":
            authoritative.append(sc)
        elif tier == "internal_guidance":
            internal.append(sc)
        elif tier == "superseded":
            superseded.append(sc)
        else:
            excluded.append(sc)

    # Case: No authoritative sources at all
    if not authoritative:
        if superseded:
            return PrecedenceDecision(
                result=PrecedenceResult.SUPERSEDED_ONLY,
                authoritative_chunks=[],
                conflicting_chunks=[],
                excluded_chunks=excluded + superseded,
                explanation="Only superseded/legacy documents were found. Current policy may differ.",
            )
        return PrecedenceDecision(
            result=PrecedenceResult.INSUFFICIENT,
            authoritative_chunks=[],
            conflicting_chunks=[],
            excluded_chunks=excluded,
            explanation="No authoritative sources found for this topic.",
        )

    # Case: Check for conflicts among authoritative sources
    # Group by source document
    docs_by_file: dict[str, list[ScoredChunk]] = {}
    for sc in authoritative:
        docs_by_file.setdefault(sc.chunk.source_file, []).append(sc)

    # If chunks come from multiple authoritative documents, check for
    # supersession relationships first
    if len(docs_by_file) > 1:
        conflict_groups = _check_for_conflicts(docs_by_file)
        relevant_groups = []
        irrelevant_conflict_chunks = []
        for group in conflict_groups:
            group_files = {sc.chunk.source_file for sc in group}
            if _conflict_relevant_to_query(group_files, query):
                relevant_groups.append(group)
            else:
                irrelevant_conflict_chunks.extend(group)

        if relevant_groups:
            all_conflicting = []
            for group in relevant_groups:
                all_conflicting.extend(group)
            return PrecedenceDecision(
                result=PrecedenceResult.CONFLICT,
                authoritative_chunks=[],
                conflicting_chunks=all_conflicting,
                excluded_chunks=excluded + superseded + irrelevant_conflict_chunks,
                explanation="Multiple active authoritative sources provide conflicting information.",
            )

        if irrelevant_conflict_chunks:
            # Conflict exists but is unrelated to the query: set aside only the
            # chunks that are actually about the conflict subject and answer
            # from the remaining authoritative sources (other sections of the
            # same files may still be relevant, e.g. packing-cube care inside
            # the product-care document).
            conflict_files = {sc.chunk.source_file for sc in irrelevant_conflict_chunks}
            entity_terms = _CONFLICT_ENTITY_TERMS.get(frozenset(conflict_files), ())
            removed = set()
            for sc in irrelevant_conflict_chunks:
                subject_text = f"{sc.chunk.heading} {sc.chunk.text}".lower()
                if any(t in subject_text for t in entity_terms):
                    excluded.append(sc)
                    removed.add(id(sc))
            # Chunks not about the conflict subject stay in `authoritative`
            # (they were never removed); drop only the subject chunks.
            authoritative = [sc for sc in authoritative if id(sc) not in removed]
            docs_by_file = {}
            for sc in authoritative:
                docs_by_file.setdefault(sc.chunk.source_file, []).append(sc)
            if not authoritative:
                return PrecedenceDecision(
                    result=PrecedenceResult.INSUFFICIENT,
                    authoritative_chunks=[],
                    conflicting_chunks=[],
                    excluded_chunks=excluded + superseded,
                    explanation="Only content unrelated to the question was found in the knowledge base.",
                )

    # No conflict detected — all authoritative chunks can be used
    # Sort by score (highest first)
    authoritative.sort(key=lambda sc: sc.score, reverse=True)

    return PrecedenceDecision(
        result=PrecedenceResult.AUTHORITATIVE,
        authoritative_chunks=authoritative,
        conflicting_chunks=[],
        excluded_chunks=excluded + superseded + internal,
        explanation=f"Found {len(authoritative)} authoritative chunk(s) from {len(docs_by_file)} source(s).",
    )


def _check_for_conflicts(docs_by_file: dict[str, list[ScoredChunk]]) -> list[list[ScoredChunk]]:
    """Check if multiple authoritative source documents genuinely conflict.

    A genuine conflict exists when two active/official documents make contradictory
    claims about the same specific entity, product, or operation without a supersession relationship.

    In the Aster & Row knowledge base, this occurs between:
    - 11-product-care.md and 12-breeze-tumbler-product-card.md (Breeze Tumbler dishwasher safety)

    Distinct policy scopes (e.g. US vs Canada shipping, Standard vs TrailPlus returns,
    Bags vs Apparel warranties) are complementary, NOT conflicts.
    """
    files = list(docs_by_file.keys())
    conflict_groups = []

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            file_a = files[i]
            file_b = files[j]
            chunks_a = docs_by_file[file_a]
            chunks_b = docs_by_file[file_b]

            # Check if there's a supersession relationship
            meta_a = chunks_a[0].chunk.metadata
            meta_b = chunks_b[0].chunk.metadata

            # If one supersedes the other, precedence resolves it
            if (meta_a.supersedes == meta_b.document_id or
                meta_b.supersedes == meta_a.document_id or
                meta_a.superseded_by == meta_b.document_id or
                meta_b.superseded_by == meta_a.document_id):
                continue

            # Only evaluate conflicts if both source documents have relevant scores
            score_a = max(c.score for c in chunks_a)
            score_b = max(c.score for c in chunks_b)
            if score_a < 0.45 or score_b < 0.45:
                continue

            # Check for substantive contradictions on the same specific product/operation
            text_a = " ".join(f"{c.chunk.source_file} {c.chunk.heading} {c.chunk.text}".lower() for c in chunks_a)
            text_b = " ".join(f"{c.chunk.source_file} {c.chunk.heading} {c.chunk.text}".lower() for c in chunks_b)

            # Breeze Tumbler dishwasher safety conflict
            if ("breeze" in text_a or "tumbler" in text_a) and ("breeze" in text_b or "tumbler" in text_b):
                if ("dishwasher" in text_a or "hand-wash" in text_a or "care" in text_a or "clean" in text_a) and \
                   ("dishwasher" in text_b or "hand-wash" in text_b or "care" in text_b or "clean" in text_b):
                    conflict_groups.append(chunks_a + chunks_b)

    return conflict_groups


def filter_by_supersession(scored_chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Remove chunks from superseded documents when the superseding
    document is also present in the results.

    This handles the case where both current and legacy returns policy
    are retrieved — the legacy one should be dropped.
    """
    # Build supersession map
    superseded_ids = set()
    active_ids = set()

    for sc in scored_chunks:
        meta = sc.chunk.metadata
        if meta.supersedes:
            active_ids.add(meta.document_id)
            superseded_ids.add(meta.supersedes)

    # If a chunk's document_id is in superseded_ids AND the superseding
    # doc is present, exclude it
    filtered = []
    for sc in scored_chunks:
        doc_id = sc.chunk.metadata.document_id
        if doc_id in superseded_ids and any(
            other.chunk.metadata.supersedes == doc_id
            for other in scored_chunks
        ):
            continue
        filtered.append(sc)

    return filtered
