import os
from dataclasses import dataclass, field

from loguru import logger
from openai import OpenAI

from src.retrieval.retriever import RetrievedChunk

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o")

SYSTEM_PROMPT = """You are a research assistant with access to a curated set of scientific papers.

Answer the user's question using ONLY the context provided below. Do not use any outside knowledge.

If the context does not contain enough information to answer the question, respond with exactly:
"I don't have enough information in the indexed papers to answer that."

When you do answer, be specific and reference which source each piece of information comes from."""

NO_ANSWER = "I don't have enough information in the indexed papers to answer that."


@dataclass
class GenerationResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    chunks_used: int = 0


def generate(question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    if not chunks:
        logger.warning("No chunks provided. Returning fallback response.")
        return GenerationResult(answer=NO_ANSWER, sources=[], chunks_used=0)

    context = _build_context(chunks)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content or NO_ANSWER
    sources = list(dict.fromkeys(c.source_file for c in chunks))

    logger.info(f"Generated answer using {len(chunks)} chunks from {len(sources)} source(s)")
    return GenerationResult(answer=answer, sources=sources, chunks_used=len(chunks))


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk.source_file}]\n{chunk.parent_text}")
    return "\n\n---\n\n".join(parts)
