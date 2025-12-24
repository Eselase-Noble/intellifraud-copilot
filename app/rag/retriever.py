from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

from app.utils.config import settings
from app.utils.logger import get_logger
from app.rag.chunking import simple_structural_chunk, Chunk
from app.rag.embeddings import embed_texts
from app.rag.reranker import rerank

logger = get_logger(__name__)

class PolicyIndex:
    def __init__(self):
        self.index_dir = Path(settings.policy_index_dir)
        self.data_dir = Path(settings.policy_data_dir)
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "chunks.jsonl"

        self._index = None
        self._chunks: List[Dict[str, Any]] = []

    def _load_policy_files(self) -> List[Tuple[str, str]]:
        files = []
        for p in sorted(self.data_dir.glob("*")):
            if p.suffix.lower() in [".md", ".txt"]:
                files.append((p.name, p.read_text(encoding="utf-8")))
            elif p.suffix.lower() == ".pdf":
                # Optional: keep pdf support minimal; seed data uses txt/md by default.
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(str(p))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    files.append((p.name, text))
                except Exception as e:
                    logger.warning("Skipping pdf %s (%s)", p.name, e)
        return files

    def build_or_load(self, force: bool = False) -> List[Dict[str, Any]]:
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if not force and self.index_path.exists() and self.meta_path.exists():
            self._load()
            return self._chunks

        logger.info("Building policy index...")
        raw_docs = self._load_policy_files()
        chunks: List[Chunk] = []
        for doc_id, text in raw_docs:
            chunks.extend(simple_structural_chunk(doc_id=doc_id, raw_text=text))

        if not chunks:
            raise RuntimeError(f"No policy documents found in {self.data_dir} (expected .txt/.md/.pdf)")

        texts = [c.text for c in chunks]
        vecs = np.array(embed_texts(texts), dtype="float32")
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        # normalize for cosine via inner product
        faiss.normalize_L2(vecs)
        index.add(vecs)

        with open(self.meta_path, "w", encoding="utf-8") as f:
            for i, c in enumerate(chunks):
                rec = {"chunk_id": i, "doc_id": c.doc_id, "section": c.section, "text": c.text}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        faiss.write_index(index, str(self.index_path))
        self._index = index
        self._chunks = [{"chunk_id": i, "doc_id": c.doc_id, "section": c.section, "text": c.text} for i, c in enumerate(chunks)]
        logger.info("Indexed %d chunks", len(self._chunks))
        return self._chunks

    def _load(self) -> None:
        self._index = faiss.read_index(str(self.index_path))
        self._chunks = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self._chunks.append(json.loads(line))
        logger.info("Loaded policy index with %d chunks", len(self._chunks))

    def query(self, q: str, top_k: int, rerank_k: int) -> List[Dict[str, Any]]:
        if self._index is None or not self._chunks:
            self.build_or_load(force=False)

        q_vec = np.array(embed_texts([q]), dtype="float32")
        faiss.normalize_L2(q_vec)
        scores, idxs = self._index.search(q_vec, top_k)
        idxs = idxs[0].tolist()

        candidates = [self._chunks[i] for i in idxs if i >= 0]
        cand_vecs = np.zeros((len(candidates), self._index.d), dtype="float32")
        # recompute candidate vectors (cheap enough for MVP); production would store them
        if candidates:
            cand_vecs = np.array(embed_texts([c["text"] for c in candidates]), dtype="float32")
            faiss.normalize_L2(cand_vecs)

        reranked = rerank(q_vec[0], cand_vecs, candidates, top_k=min(rerank_k, len(candidates)))
        return reranked
