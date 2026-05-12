from unittest.mock import MagicMock, patch

import pytest

from src.generation.generator import NO_ANSWER, GenerationResult, _build_context, generate
from src.retrieval.retriever import RetrievedChunk


@pytest.fixture(autouse=True)
def set_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def make_chunk(
    source_file: str = "kim_2020.pdf",
    parent_text: str = "Full parent section text.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk_0",
        doc_id="kim_2020",
        source_file=source_file,
        child_text="short child text",
        parent_text=parent_text,
        score=0.9,
    )


@patch("src.generation.generator.OpenAI")
def test_generate_returns_generation_result(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="ecDNA drives oncogene amplification."))
    ]
    result = generate("What is ecDNA?", [make_chunk()])
    assert isinstance(result, GenerationResult)


@patch("src.generation.generator.OpenAI")
def test_generate_returns_answer_text(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="ecDNA drives oncogene amplification."))
    ]
    result = generate("What is ecDNA?", [make_chunk()])
    assert result.answer == "ecDNA drives oncogene amplification."


@patch("src.generation.generator.OpenAI")
def test_generate_extracts_sources(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Some answer."))
    ]
    chunks = [make_chunk("paper_a.pdf"), make_chunk("paper_b.pdf"), make_chunk("paper_a.pdf")]
    result = generate("question", chunks)
    # sources should be deduplicated and order-preserving
    assert result.sources == ["paper_a.pdf", "paper_b.pdf"]


@patch("src.generation.generator.OpenAI")
def test_generate_reports_chunks_used(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Answer."))
    ]
    chunks = [make_chunk() for _ in range(5)]
    result = generate("question", chunks)
    assert result.chunks_used == 5


def test_generate_returns_fallback_when_no_chunks():
    result = generate("What is ecDNA?", [])
    assert result.answer == NO_ANSWER
    assert result.sources == []
    assert result.chunks_used == 0


def test_build_context_uses_parent_text():
    chunk = make_chunk(parent_text="This is the full parent section.")
    context = _build_context([chunk])
    assert "This is the full parent section." in context


def test_build_context_includes_source_label():
    chunk = make_chunk(source_file="kim_2020.pdf")
    context = _build_context([chunk])
    assert "kim_2020.pdf" in context


def test_build_context_separates_multiple_chunks():
    chunks = [make_chunk(f"paper_{i}.pdf") for i in range(3)]
    context = _build_context(chunks)
    assert "---" in context
