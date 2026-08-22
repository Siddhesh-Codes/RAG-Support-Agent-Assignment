"""Knowledge base ingestion: parse markdown, extract front matter, chunk.

Design decisions:
- Chunks split on markdown headings (## sections) to preserve semantic coherence.
- Each chunk keeps full front-matter metadata so precedence engine can work per-chunk.
- Front matter is parsed but NOT included in chunk text (it's metadata, not content).
- Chunks include section heading for context and citation.
"""

import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DocumentMetadata:
    """Parsed front-matter metadata from a knowledge base document."""
    document_id: str = ""
    title: str = ""
    status: str = ""             # active, superseded, draft
    effective_date: str = ""
    last_reviewed: str = ""
    audience: str = ""           # customer, internal
    policy_authority: str = ""   # official, none
    supersedes: str = ""         # document_id this doc replaces
    superseded_by: str = ""      # document_id that replaces this doc
    superseded_date: str = ""
    customer_answering: Optional[bool] = None  # explicit flag on some docs

    @property
    def is_authoritative(self) -> bool:
        """Is this document an active, official, customer-facing policy?"""
        return (
            self.status == "active"
            and self.policy_authority == "official"
        )

    @property
    def is_superseded(self) -> bool:
        return self.status == "superseded"

    @property
    def is_draft(self) -> bool:
        return self.status == "draft"

    @property
    def is_internal(self) -> bool:
        return self.audience == "internal"

    @property
    def is_customer_facing(self) -> bool:
        return self.audience == "customer"

    @property
    def has_no_authority(self) -> bool:
        return self.policy_authority == "none"


@dataclass
class Chunk:
    """A single retrievable unit of knowledge base content."""
    text: str                     # The actual content (no front-matter YAML)
    source_file: str              # Filename, e.g. "01-returns-policy-current.md"
    heading: str                  # Section heading, e.g. "Standard return window"
    metadata: DocumentMetadata    # Full parsed front-matter
    chunk_index: int = 0          # Position within document

    def citation_label(self) -> str:
        """Human-readable citation string."""
        parts = [self.source_file]
        if self.heading:
            parts.append(f"- {self.heading}")
        return " ".join(parts)


def parse_front_matter(content: str) -> tuple[DocumentMetadata, str]:
    """Extract YAML front matter and return (metadata, body).

    Handles the standard --- delimited front matter at the start of markdown.
    """
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not fm_match:
        return DocumentMetadata(), content

    try:
        raw = yaml.safe_load(fm_match.group(1))
        if not isinstance(raw, dict):
            return DocumentMetadata(), content
    except yaml.YAMLError:
        return DocumentMetadata(), content

    meta = DocumentMetadata(
        document_id=str(raw.get("document_id", "")),
        title=str(raw.get("title", "")),
        status=str(raw.get("status", "")),
        effective_date=str(raw.get("effective_date", "")),
        last_reviewed=str(raw.get("last_reviewed", "")),
        audience=str(raw.get("audience", "")),
        policy_authority=str(raw.get("policy_authority", "")),
        supersedes=str(raw.get("supersedes", "")),
        superseded_by=str(raw.get("superseded_by", "")),
        superseded_date=str(raw.get("superseded_date", "")),
        customer_answering=raw.get("customer_answering"),
    )

    body = content[fm_match.end():]
    return meta, body


def chunk_document(body: str, source_file: str, metadata: DocumentMetadata) -> list[Chunk]:
    """Split a document body into chunks based on ## headings.

    Strategy:
    - Split on ## headings to get semantically coherent sections.
    - The document title (# heading) is combined with the first section.
    - Each chunk includes the document title as context prefix.
    - Very short chunks (< 20 chars of content) are merged with the next.
    """
    # Extract document-level title (# heading)
    title_match = re.match(r"^#\s+(.+?)$", body.strip(), re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else metadata.title

    # Split on ## headings
    sections = re.split(r"(?=^##\s+)", body.strip(), flags=re.MULTILINE)

    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # Extract section heading
        heading_match = re.match(r"^##\s+(.+?)$", section, re.MULTILINE)
        heading = heading_match.group(1).strip() if heading_match else doc_title

        # Skip chunks that are just the title with no content
        content_without_headings = re.sub(r"^#+\s+.*$", "", section, flags=re.MULTILINE).strip()
        if len(content_without_headings) < 20 and i < len(sections) - 1:
            # Merge tiny intro sections into next section
            if i + 1 < len(sections):
                sections[i + 1] = section + "\n\n" + sections[i + 1]
            continue

        # Prefix with document title for context (helps retrieval)
        chunk_text = f"[{doc_title}]\n\n{section}"

        chunks.append(Chunk(
            text=chunk_text,
            source_file=source_file,
            heading=heading,
            metadata=metadata,
            chunk_index=len(chunks),
        ))

    # If no chunks were created (edge case), make one from the whole body
    if not chunks and body.strip():
        chunks.append(Chunk(
            text=f"[{doc_title}]\n\n{body.strip()}",
            source_file=source_file,
            heading=doc_title,
            metadata=metadata,
            chunk_index=0,
        ))

    return chunks


def ingest_knowledge_base(kb_dir: Path) -> list[Chunk]:
    """Parse and chunk all markdown files in the knowledge base directory.

    Returns all chunks with their metadata. Does NOT filter or rank —
    that's the precedence engine's job.
    """
    all_chunks = []

    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {kb_dir}")

    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8")
        metadata, body = parse_front_matter(content)
        source_file = md_path.name

        chunks = chunk_document(body, source_file, metadata)
        all_chunks.extend(chunks)

    return all_chunks
