"""Document ingestion: load MGC Markdown files and split them into
structure-aware chunks with rich metadata.

The supplied documents are Markdown, which already carries structure (headings,
tables, lists). We split on headings so each chunk represents a meaningful
section, and we attach metadata that lets the retriever and the answerer cite
the exact source file and section.

No text is invented here: we only split and annotate the supplied text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

def _document_type(source_file: str) -> str:
    """Infer a coarse document type from the filename."""
    name = Path(source_file).name.lower()
    if "price" in name or "payment" in name:
        return "price_list"
    if "policy" in name or "faq" in name:
        return "booking_policy"
    if "brochure" in name:
        return "brochure"
    return "general"


def _document_name(source_file: str) -> str:
    """A clean, human-readable name derived from the source filename."""
    return Path(source_file).stem.replace("_", " ")


def _split_heading_blocks(text: str) -> list[tuple[str, str]]:
    """Split Markdown text into (heading, body) blocks by '#' headings.

    The first block (before any heading) uses an empty heading marker.
    Returns a list of (heading, block_text) pairs.
    """
    pattern = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
    blocks: list[tuple[str, str]] = []
    positions = [(m.start(), m.group(0)) for m in pattern.finditer(text)]

    for idx, (pos, heading_line) in enumerate(positions):
        # Determine block boundaries.
        start = pos
        if idx + 1 < len(positions):
            end = positions[idx + 1][0]
        else:
            end = len(text)
        body = text[start:end]
        heading = heading_line.lstrip("#").strip()
        blocks.append((heading, body))

    # Text appearing before the first heading belongs to a preamble block.
    if positions:
        preamble = text[: positions[0][0]].strip()
        if preamble:
            blocks.insert(0, ("Preamble", preamble))
    elif text.strip():
        blocks.append(("Preamble", text.strip()))

    return blocks


def _chunk_block(heading: str, body: str, chunk_size: int = 1800) -> list[str]:
    """Split one section's body into chunks no longer than ``chunk_size``.

    Sections that fit within the limit are kept whole so that related facts
    (e.g. a price table and its adjacent premiums) stay together. Oversized
    sections are split on sentence boundaries to avoid cutting mid-thought.
    """
    if len(body) <= chunk_size:
        return [body]

    sentences = re.split(r"(?<=[.!?])\s+", body)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > chunk_size and current:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


@dataclass
class LoadedDocument:
    """Raw loaded Markdown content plus its metadata."""

    source_file: str
    text: str


def _load_files(docs_dir: Path) -> list[LoadedDocument]:
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_dir}. "
            "Expected the MGC Markdown files under the configured docs dir."
        )
    loaded: list[LoadedDocument] = []
    for path in sorted(docs_dir.glob("*.md")):
        loaded.append(
            LoadedDocument(
                source_file=path.name,
                text=path.read_text(encoding="utf-8"),
            )
        )
    if not loaded:
        raise FileNotFoundError(
            f"No Markdown documents found in {docs_dir}. "
            "Add the MGC .md files and run build_index.py."
        )
    return loaded


def load_and_chunk_documents(
    docs_dir: Path,
    chunk_id_prefix: str = "mgc",
    chunk_size: int = 1800,
) -> list[Document]:
    """Load all Markdown docs in ``docs_dir`` and return structured chunks.

    Each chunk is a LangChain Document whose ``page_content`` is the raw text
    and whose ``metadata`` records the source file, document name, section
    heading, chunk id and inferred document type.
    """
    documents: list[Document] = []
    for doc in _load_files(docs_dir):
        source_file = doc.source_file
        d_type = _document_type(source_file)
        d_name = _document_name(source_file)
        raw_id = _stem_id(source_file)

        for section_index, (heading, body) in enumerate(
            _split_heading_blocks(doc.text)
        ):
            for part_index, chunk_text in enumerate(
                _chunk_block(heading, body, chunk_size=chunk_size)
            ):
                chunk_id = (
                    f"{chunk_id_prefix}-{raw_id}-s{section_index}-p{part_index}"
                )
                metadata = {
                    "source_file": source_file,
                    "document_name": d_name,
                    "section": heading,
                    "chunk_id": chunk_id,
                    "document_type": d_type,
                    # Prepend the heading so a chunk is self-describing when
                    # retrieved (eases both relevance checks and citation).
                    "header": f"# {heading}" if heading else "",
                }
                content = chunk_text
                documents.append(Document(page_content=content, metadata=metadata))
    return documents


def _stem_id(source_file: str) -> str:
    """A short stable id from the filename, e.g. '02' from '02_price...'."""
    match = re.match(r"(\d+)", Path(source_file).stem)
    return match.group(1) if match else Path(source_file).stem


def format_evidence(docs: Iterable[Document]) -> str:
    """Render retrieved documents as a compact evidence block for the LLM."""
    parts: list[str] = []
    for doc in docs:
        meta = doc.metadata
        parts.append(
            f"[Source: {meta.get('source_file')} | Section: {meta.get('section')}]\n"
            f"{doc.page_content.strip()}\n"
        )
    return "\n".join(parts)
