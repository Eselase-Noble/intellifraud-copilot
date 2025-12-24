from __future__ import annotations
import pytest
from app.rag.chunking import simple_structural_chunk

def test_chunker_produces_chunks():
    text = "# Title\nSome content here.\n\n## Sub\nMore content."
    chunks = simple_structural_chunk("doc.txt", text, max_chars=40, overlap=5)
    assert len(chunks) >= 1
