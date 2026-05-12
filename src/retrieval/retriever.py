import os
from dataclasses import dataclass

from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.embedding.embedder import embed_query
from src.vectorstore.store import COLLECTION_NAME, get_client, search

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Standard constant for RRF. Higher k = less aggressive fusion.
# 60 is the widely used default from the original RRF paper.
RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_file: str
    child_text: str
    parent_text: str
    score: float


class Retriever:
    def __init__(self):
        logger.info("Initializing retriever...")
        self._all_chunks = self._fetch_all_chunks()
        self._bm25 = self._build_bm25_index(self._all_chunks)
        self._reranker = CrossEncoder(RERANKER_MODEL)
        logger.info(f"Retriever ready. {len(self._all_chunks)} chunks indexed for BM25.")

    def _fetch_all_chunks(self) -> list[dict]:
        """Pull all child chunks from Qdrant to build the BM25 index in memory.

        We do this once at startup. BM25 needs the full text of every chunk
        to compute term frequencies. Qdrant does not do keyword search natively,
        so we handle it here.
        """
        client = get_client()
        all_points = []
        offset = None

        while True:
            results, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(results)
            if next_offset is None:
                break
            offset = next_offset

        logger.info(f"Fetched {len(all_points)} chunks from Qdrant for BM25 index")
        return [{"id": p.id, **p.payload} for p in all_points]

    def _build_bm25_index(self, chunks: list[dict]) -> BM25Okapi:
        tokenized = [chunk["child_text"].lower().split() for chunk in chunks]
        return BM25Okapi(tokenized)

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {**self._all_chunks[i], "score": float(scores[i])}
            for i in top_indices
        ]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Merge two ranked lists without needing to normalize their scores.

        Formula: rrf_score = 1/(k + rank_dense) + 1/(k + rank_bm25)
        A chunk appearing high in both lists scores highest.
        A chunk appearing in only one list gets partial credit.
        """
        scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for rank, chunk in enumerate(dense_results):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
            chunk_data[cid] = chunk

        for rank, chunk in enumerate(bm25_results):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
            chunk_data[cid] = chunk

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
        return [{**chunk_data[cid], "rrf_score": scores[cid]} for cid in sorted_ids]

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[RetrievedChunk]:
        """Score each (question, chunk) pair jointly using a cross-encoder.

        Unlike the embedding model which scores query and chunk independently,
        the cross-encoder reads them together and produces a more accurate
        relevance score. Slower, so we only run it on the top 20 fused candidates.
        """
        pairs = [(query, c["child_text"]) for c in candidates]
        cross_scores = self._reranker.predict(pairs)

        ranked = sorted(zip(cross_scores, candidates), key=lambda x: x[0], reverse=True)[:top_k]

        return [
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                source_file=c["source_file"],
                child_text=c["child_text"],
                parent_text=c["parent_text"],
                score=float(score),
            )
            for score, c in ranked
        ]

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        logger.info(f"Query: '{question[:80]}'")

        query_vector = embed_query(question)

        dense_results = search(query_vector, top_k=50)
        logger.info(f"Dense search: {len(dense_results)} candidates")

        bm25_results = self._bm25_search(question, top_k=50)
        logger.info(f"BM25 search: {len(bm25_results)} candidates")

        fused = self._reciprocal_rank_fusion(dense_results, bm25_results, top_k=20)
        logger.info(f"After RRF fusion: {len(fused)} candidates")

        reranked = self._rerank(question, fused, top_k=top_k)
        logger.info(f"After reranking: {len(reranked)} chunks returned")

        return reranked
