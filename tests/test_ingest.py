"""Tests for knowledge base ingestion: front-matter parsing and chunking."""

import pytest
from pathlib import Path
from src.rag.ingest import (
    parse_front_matter,
    chunk_document,
    ingest_knowledge_base,
    DocumentMetadata,
)


# --- Front matter parsing ---

class TestParseFrontMatter:
    def test_parses_standard_front_matter(self):
        content = """---
document_id: RET-2026-01
title: Returns Policy
status: active
effective_date: 2026-04-01
audience: customer
policy_authority: official
---

# Returns Policy

Some content here.
"""
        meta, body = parse_front_matter(content)
        assert meta.document_id == "RET-2026-01"
        assert meta.title == "Returns Policy"
        assert meta.status == "active"
        assert meta.audience == "customer"
        assert meta.policy_authority == "official"
        assert "# Returns Policy" in body
        assert "---" not in body.strip().split("\n")[0]  # No front matter in body

    def test_parses_supersession_metadata(self):
        content = """---
document_id: RET-2024-01
status: superseded
superseded_by: RET-2026-01
superseded_date: 2026-04-01
audience: customer
policy_authority: official
---

Body text.
"""
        meta, body = parse_front_matter(content)
        assert meta.is_superseded
        assert meta.superseded_by == "RET-2026-01"
        assert not meta.is_authoritative

    def test_parses_draft_document(self):
        content = """---
document_id: MIG-TEST-04
status: draft
audience: internal
policy_authority: none
customer_answering: false
---

Draft content.
"""
        meta, body = parse_front_matter(content)
        assert meta.is_draft
        assert meta.is_internal
        assert meta.has_no_authority
        assert meta.customer_answering is False
        assert not meta.is_authoritative

    def test_no_front_matter(self):
        content = "# Just a heading\n\nSome text."
        meta, body = parse_front_matter(content)
        assert meta.document_id == ""
        assert body == content

    def test_malformed_yaml(self):
        content = "---\n: invalid: yaml:\n---\n\nBody."
        meta, body = parse_front_matter(content)
        # Should not crash — returns empty metadata
        assert isinstance(meta, DocumentMetadata)

    def test_metadata_properties(self):
        # Authoritative: active + official
        meta = DocumentMetadata(status="active", policy_authority="official", audience="customer")
        assert meta.is_authoritative
        assert meta.is_customer_facing
        assert not meta.is_superseded
        assert not meta.is_draft
        assert not meta.is_internal
        assert not meta.has_no_authority

        # Internal
        meta_int = DocumentMetadata(status="active", policy_authority="official", audience="internal")
        assert meta_int.is_authoritative  # it's active+official
        assert meta_int.is_internal


# --- Chunking ---

class TestChunking:
    def test_chunks_on_h2_headings(self):
        meta = DocumentMetadata(document_id="TEST-01", title="Test Doc", status="active")
        body = """# Test Doc

## Section One

Content for section one is substantial enough to be a full chunk on its own.

## Section Two

Content for section two is also substantial enough to be a standalone chunk.
"""
        chunks = chunk_document(body, "test.md", meta)
        assert len(chunks) >= 2
        # Section two should definitely have its heading preserved
        headings = [c.heading for c in chunks]
        assert "Section Two" in headings
        # The first chunk may have merged the title intro into section one,
        # so section one's content should appear somewhere in the chunks
        all_text = " ".join(c.text for c in chunks)
        assert "section one" in all_text.lower()

    def test_preserves_metadata_on_chunks(self):
        meta = DocumentMetadata(document_id="RET-2026-01", status="active",
                                policy_authority="official", audience="customer")
        body = "# Title\n\n## Section\n\nContent here that is long enough to be a real chunk."
        chunks = chunk_document(body, "01-returns.md", meta)
        assert all(c.metadata.document_id == "RET-2026-01" for c in chunks)
        assert all(c.source_file == "01-returns.md" for c in chunks)

    def test_citation_label(self):
        meta = DocumentMetadata()
        body = "# Doc Title\n\n## My Section\n\nSome content that is a meaningful passage."
        chunks = chunk_document(body, "file.md", meta)
        for c in chunks:
            label = c.citation_label()
            assert "file.md" in label

    def test_empty_body(self):
        meta = DocumentMetadata()
        chunks = chunk_document("", "empty.md", meta)
        assert chunks == []

    def test_no_h2_makes_single_chunk(self):
        meta = DocumentMetadata(title="Simple")
        body = "# Simple\n\nJust one paragraph of content that should create a single chunk."
        chunks = chunk_document(body, "simple.md", meta)
        assert len(chunks) == 1


# --- Full ingestion ---

class TestIngestion:
    def test_ingest_real_knowledge_base(self):
        """Integration test: parse the actual knowledge base."""
        kb_dir = Path(__file__).resolve().parent.parent / "knowledge-base"
        if not kb_dir.exists():
            pytest.skip("Knowledge base not found")

        chunks = ingest_knowledge_base(kb_dir)

        # Should have chunks from all 14 documents
        source_files = set(c.source_file for c in chunks)
        assert "01-returns-policy-current.md" in source_files
        assert "14-internal-content-migration-notes.md" in source_files
        assert len(chunks) >= 14  # At least one chunk per doc

        # Verify metadata is preserved
        current_returns = [c for c in chunks if c.source_file == "01-returns-policy-current.md"]
        assert all(c.metadata.status == "active" for c in current_returns)
        assert all(c.metadata.document_id == "RET-2026-01" for c in current_returns)

        legacy_returns = [c for c in chunks if c.source_file == "02-returns-policy-legacy.md"]
        assert all(c.metadata.status == "superseded" for c in legacy_returns)

        draft = [c for c in chunks if c.source_file == "14-internal-content-migration-notes.md"]
        assert all(c.metadata.status == "draft" for c in draft)
        assert all(c.metadata.policy_authority == "none" for c in draft)
