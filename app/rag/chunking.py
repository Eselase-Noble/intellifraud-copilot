from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import re

@dataclass
class Chunk:
    doc_id: str
    section: str
    text: str

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def simple_structural_chunk(doc_id: str, raw_text: str, max_chars: int = 900, overlap: int = 120) -> List[Chunk]:
    """
    A practical chunker:
    - splits on headings (lines starting with '#', '##', etc.) when present
    - otherwise falls back to sliding windows
    """
    raw_text = raw_text.replace("\r\n", "\n")
    blocks = []
    current_title = "root"
    buf = []
    for line in raw_text.split("\n"):
        if line.strip().startswith("#"):
            if buf:
                blocks.append((current_title, "\n".join(buf)))
                buf = []
            current_title = normalize_ws(line.strip("# ").strip()) or "section"
        else:
            buf.append(line)
    if buf:
        blocks.append((current_title, "\n".join(buf)))

    chunks: List[Chunk] = []
    for title, block in blocks:
        text = normalize_ws(block)
        if not text:
            continue
        if len(text) <= max_chars:
            chunks.append(Chunk(doc_id=doc_id, section=title, text=text))
            continue

        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            piece = text[start:end]
            chunks.append(Chunk(doc_id=doc_id, section=title, text=piece))
            if end == len(text):
                break
            start = max(0, end - overlap)

    return chunks
