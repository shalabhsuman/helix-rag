from unittest.mock import MagicMock, patch

from src.models import ChildChunk, ParentChunk
from src.vectorstore.store import delete_by_doc_id, search, upsert_chunks


def make_parent(idx: int) -> ParentChunk:
    return ParentChunk(
        chunk_id=f"doc_parent_{idx}",
        doc_id="doc",
        source_file="doc.pdf",
        text=f"Parent text {idx}",
        chunk_index=idx,
    )


def make_child(idx: int) -> ChildChunk:
    return ChildChunk(
        chunk_id=f"doc_child_{idx}",
        parent_chunk_id=f"doc_parent_{idx}",
        doc_id="doc",
        source_file="doc.pdf",
        text=f"Child text {idx}",
        chunk_index=idx,
    )


@patch("src.vectorstore.store.get_client")
def test_upsert_calls_qdrant_upsert(mock_get_client):
    mock_client = mock_get_client.return_value
    mock_client.get_collections.return_value.collections = []

    children = [make_child(i) for i in range(3)]
    parents = [make_parent(i) for i in range(3)]
    vectors = [[0.1] * 1536 for _ in range(3)]

    upsert_chunks(children, parents, vectors)

    mock_client.upsert.assert_called_once()
    call_kwargs = mock_client.upsert.call_args.kwargs
    assert len(call_kwargs["points"]) == 3


@patch("src.vectorstore.store.get_client")
def test_upsert_payload_contains_parent_text(mock_get_client):
    mock_client = mock_get_client.return_value
    mock_client.get_collections.return_value.collections = []

    children = [make_child(0)]
    parents = [make_parent(0)]
    vectors = [[0.1] * 1536]

    upsert_chunks(children, parents, vectors)

    point = mock_client.upsert.call_args.kwargs["points"][0]
    assert point.payload["parent_text"] == "Parent text 0"
    assert point.payload["child_text"] == "Child text 0"
    assert point.payload["doc_id"] == "doc"


@patch("src.vectorstore.store.get_client")
def test_search_returns_list_of_dicts(mock_get_client):
    mock_result = MagicMock()
    mock_result.score = 0.9
    mock_result.payload = {
        "doc_id": "doc",
        "child_text": "some text",
        "parent_text": "full section",
    }
    mock_get_client.return_value.query_points.return_value.points = [mock_result]

    results = search([0.1] * 1536, top_k=5)

    assert len(results) == 1
    assert results[0]["score"] == 0.9
    assert results[0]["doc_id"] == "doc"


@patch("src.vectorstore.store.get_client")
def test_delete_calls_qdrant_delete(mock_get_client):
    delete_by_doc_id("kim_2020")
    mock_get_client.return_value.delete.assert_called_once()
