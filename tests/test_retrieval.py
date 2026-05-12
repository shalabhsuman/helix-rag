from unittest.mock import MagicMock, patch

from src.retrieval.retriever import RRF_K, RetrievedChunk, Retriever


def make_chunk(chunk_id: str, text: str = "sample text about ecDNA research") -> dict:
    return {
        "id": 0,
        "chunk_id": chunk_id,
        "parent_chunk_id": "parent_0",
        "doc_id": "test_doc",
        "source_file": "test.pdf",
        "child_text": text,
        "parent_text": "full parent section text",
        "score": 0.9,
    }


def make_retriever_with_mocks(chunks: list[dict]) -> Retriever:
    """Build a Retriever with Qdrant and CrossEncoder mocked out."""
    with (
        patch("src.retrieval.retriever.get_client") as mock_client,
        patch("src.retrieval.retriever.CrossEncoder") as mock_ce,
    ):
        mock_scroll = MagicMock()
        mock_scroll.return_value = (
            [MagicMock(id=i, payload=c) for i, c in enumerate(chunks)],
            None,
        )
        mock_client.return_value.scroll = mock_scroll
        mock_ce.return_value = MagicMock()
        retriever = Retriever()
    return retriever


def test_bm25_returns_results():
    chunks = [make_chunk(f"chunk_{i}", f"ecDNA oncogene amplification text {i}") for i in range(5)]
    retriever = make_retriever_with_mocks(chunks)
    results = retriever._bm25_search("ecDNA oncogene", top_k=3)
    assert len(results) == 3


def test_bm25_respects_top_k():
    chunks = [make_chunk(f"chunk_{i}") for i in range(10)]
    retriever = make_retriever_with_mocks(chunks)
    results = retriever._bm25_search("ecDNA", top_k=4)
    assert len(results) == 4


def test_rrf_boosts_chunk_appearing_in_both_lists():
    chunks = [make_chunk(f"chunk_{i}") for i in range(5)]
    retriever = make_retriever_with_mocks(chunks)

    # chunk_A appears #1 in dense and #1 in BM25
    # chunk_B appears #2 in dense only
    chunk_a = make_chunk("chunk_A")
    chunk_b = make_chunk("chunk_B")
    chunk_c = make_chunk("chunk_C")

    dense = [chunk_a, chunk_b, chunk_c]
    bm25 = [chunk_a, chunk_c, chunk_b]

    fused = retriever._reciprocal_rank_fusion(dense, bm25, top_k=3)
    top_id = fused[0]["chunk_id"]
    assert top_id == "chunk_A"


def test_rrf_scores_use_correct_formula():
    chunks = [make_chunk(f"chunk_{i}") for i in range(3)]
    retriever = make_retriever_with_mocks(chunks)

    chunk_a = make_chunk("chunk_A")
    dense = [chunk_a]
    bm25 = [chunk_a]

    fused = retriever._reciprocal_rank_fusion(dense, bm25, top_k=1)
    expected_score = 1 / (RRF_K + 1) + 1 / (RRF_K + 1)
    assert abs(fused[0]["rrf_score"] - expected_score) < 1e-9


def test_rrf_respects_top_k():
    chunks = [make_chunk(f"chunk_{i}") for i in range(5)]
    retriever = make_retriever_with_mocks(chunks)

    dense = [make_chunk(f"d_{i}") for i in range(5)]
    bm25 = [make_chunk(f"b_{i}") for i in range(5)]

    fused = retriever._reciprocal_rank_fusion(dense, bm25, top_k=3)
    assert len(fused) == 3


def test_rerank_returns_retrieved_chunks():
    chunks = [make_chunk(f"chunk_{i}") for i in range(5)]
    retriever = make_retriever_with_mocks(chunks)
    retriever._reranker.predict = MagicMock(return_value=[0.9, 0.7, 0.5, 0.3, 0.1])

    candidates = [make_chunk(f"chunk_{i}") for i in range(5)]
    results = retriever._rerank("test question", candidates, top_k=3)

    assert len(results) == 3
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_rerank_sorts_by_score_descending():
    chunks = [make_chunk(f"chunk_{i}") for i in range(3)]
    retriever = make_retriever_with_mocks(chunks)
    retriever._reranker.predict = MagicMock(return_value=[0.3, 0.9, 0.6])

    candidates = [make_chunk(f"chunk_{i}") for i in range(3)]
    results = retriever._rerank("test question", candidates, top_k=3)

    assert results[0].score > results[1].score > results[2].score
