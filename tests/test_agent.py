from unittest.mock import MagicMock, patch

import pytest

from src.agent.agent import _list_papers, _search_papers, build_agent


@pytest.fixture(autouse=True)
def set_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


# --- search_papers tool ---

@patch("src.agent.agent._get_retriever")
@patch("src.agent.agent.generate")
def test_search_papers_returns_answer(mock_generate, mock_get_retriever):
    mock_chunk = MagicMock()
    mock_chunk.parent_text = "ecDNA drives amplification."
    mock_chunk.source_file = "kim_2020.pdf"
    mock_get_retriever.return_value.retrieve.return_value = [mock_chunk]
    mock_generate.return_value = MagicMock(
        answer="ecDNA amplifies oncogenes.",
        sources=["kim_2020.pdf"],
    )
    result = _search_papers("What is ecDNA?")
    assert "ecDNA amplifies oncogenes." in result


@patch("src.agent.agent._get_retriever")
@patch("src.agent.agent.generate")
def test_search_papers_includes_sources(mock_generate, mock_get_retriever):
    mock_chunk = MagicMock()
    mock_chunk.source_file = "kim_2020.pdf"
    mock_get_retriever.return_value.retrieve.return_value = [mock_chunk]
    mock_generate.return_value = MagicMock(
        answer="Some answer.",
        sources=["kim_2020.pdf"],
    )
    result = _search_papers("What is ecDNA?")
    assert "kim_2020.pdf" in result


@patch("src.agent.agent._get_retriever")
@patch("src.agent.agent.generate")
def test_search_papers_no_chunks_returns_fallback(mock_generate, mock_get_retriever):
    mock_get_retriever.return_value.retrieve.return_value = []
    mock_generate.return_value = MagicMock(
        answer="I don't have enough information.",
        sources=[],
    )
    result = _search_papers("Unknown topic")
    assert "I don't have enough information." in result


# --- list_papers tool ---

def test_list_papers_returns_paper_names(tmp_path, monkeypatch):
    monkeypatch.setattr("src.agent.agent.RAW_DIR", tmp_path)
    (tmp_path / "kim_2020.pdf").touch()
    (tmp_path / "bailey_2024.pdf").touch()
    result = _list_papers()
    assert "kim_2020" in result
    assert "bailey_2024" in result


def test_list_papers_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("src.agent.agent.RAW_DIR", tmp_path)
    result = _list_papers()
    assert "No papers" in result


# --- agent construction ---

def test_build_agent_returns_agent():
    from agents import Agent
    agent = build_agent()
    assert isinstance(agent, Agent)


def test_build_agent_has_two_tools():
    agent = build_agent()
    assert len(agent.tools) == 2
