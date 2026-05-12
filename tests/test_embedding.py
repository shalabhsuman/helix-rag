from unittest.mock import MagicMock, patch

from src.embedding.embedder import embed_query, embed_texts


def make_mock_response(n: int, dims: int = 1536):
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1] * dims) for _ in range(n)]
    return response


@patch("src.embedding.embedder.get_client")
def test_embed_texts_returns_one_vector_per_text(mock_get_client):
    mock_get_client.return_value.embeddings.create.return_value = make_mock_response(3)
    result = embed_texts(["text one", "text two", "text three"])
    assert len(result) == 3


@patch("src.embedding.embedder.get_client")
def test_embed_texts_vector_has_correct_dimensions(mock_get_client):
    mock_get_client.return_value.embeddings.create.return_value = make_mock_response(1)
    result = embed_texts(["some text"])
    assert len(result[0]) == 1536


@patch("src.embedding.embedder.get_client")
def test_embed_query_returns_single_vector(mock_get_client):
    mock_get_client.return_value.embeddings.create.return_value = make_mock_response(1)
    result = embed_query("what is ecDNA?")
    assert isinstance(result, list)
    assert len(result) == 1536


@patch("src.embedding.embedder.get_client")
def test_embed_texts_batches_large_inputs(mock_get_client):
    # 250 texts should trigger 3 batches (100 + 100 + 50)
    mock_client = mock_get_client.return_value
    mock_client.embeddings.create.side_effect = [
        make_mock_response(100),
        make_mock_response(100),
        make_mock_response(50),
    ]
    result = embed_texts(["text"] * 250)
    assert len(result) == 250
    assert mock_client.embeddings.create.call_count == 3
