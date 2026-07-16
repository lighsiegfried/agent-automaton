"""Deterministic chunking (Phase 6A, item 3).

Each extracted block is windowed into fixed-size, overlapping chunks that inherit the
block's page/heading/section/row metadata, so every chunk can be cited precisely. The
same document + settings always yields byte-identical chunks (stable ids derived from
the document id + position), which keeps re-indexing and tests reproducible.
"""

from __future__ import annotations

from app.knowledge.models import Chunk, ExtractedDoc


def chunk_document(extracted: ExtractedDoc, *, doc_id: str, owner: str,
                   chunk_size: int, overlap: int) -> list[Chunk]:
    step = max(1, chunk_size - max(0, overlap))
    chunks: list[Chunk] = []
    offset = 0
    position = 0
    for block in extracted.blocks:
        text = (block.text or "").strip()
        if not text:
            offset += 1
            continue
        starts = [0] if len(text) <= chunk_size else list(range(0, len(text), step))
        for start in starts:
            piece = text[start:start + chunk_size].strip()
            if not piece:
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_c{position}", doc_id=doc_id, owner=owner, position=position,
                text=piece, page=block.page, heading=block.heading, section=block.section,
                row=block.row, char_start=offset + start, char_end=offset + start + len(piece)))
            position += 1
        offset += len(text) + 1
    return chunks
