"""OpenAI Agents SDK wrapper for the helix-rag pipeline.

Exposes two tools to a GPT-4o agent:
  - search_papers: runs the full RAG pipeline (retrieve + rerank + generate)
  - list_papers:   lists all PDF files currently indexed

The agent decides which tool to call based on the user's message.
Conversation history is maintained across turns within a session.
"""

import os
from pathlib import Path

from agents import Agent, function_tool
from loguru import logger

from src.generation.generator import generate
from src.retrieval.retriever import Retriever

RAW_DIR = Path("data/raw")

AGENT_INSTRUCTIONS = """You are a research assistant with access to a curated set of
indexed scientific papers.

You have two tools:
- search_papers: use this when the user asks a question about the science,
  findings, methods, or content of the papers.
- list_papers: use this when the user wants to know which papers are available.

Always cite the source paper when you answer a scientific question.
If the retrieved context does not contain enough information, say so clearly.
Do not use outside knowledge — only what the papers contain."""

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        logger.info("Initializing retriever for agent...")
        _retriever = Retriever()
    return _retriever


def _search_papers(question: str) -> str:
    """Search the indexed research papers and return a cited answer.

    Use this for any question about the scientific content, findings, methods,
    or conclusions in the papers. Returns an answer with source citations.
    """
    retriever = _get_retriever()
    chunks = retriever.retrieve(question, top_k=5)
    result = generate(question, chunks)
    if not chunks:
        return result.answer
    sources = ", ".join(result.sources)
    return f"{result.answer}\n\nSources: {sources}"


def _list_papers() -> str:
    """List all research papers currently indexed and available for search.

    Use this when the user asks what papers are available, what documents
    are indexed, or what topics are covered.
    """
    if not RAW_DIR.exists():
        return "No papers directory found."
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        return "No papers are currently indexed."
    lines = [f"{i+1}. {p.stem}" for i, p in enumerate(pdfs)]
    return "Indexed papers:\n" + "\n".join(lines)


search_papers = function_tool(_search_papers)
list_papers = function_tool(_list_papers)


def build_agent() -> Agent:
    return Agent(
        name="Research Assistant",
        model=os.getenv("GENERATION_MODEL", "gpt-4o"),
        instructions=AGENT_INSTRUCTIONS,
        tools=[search_papers, list_papers],
    )
