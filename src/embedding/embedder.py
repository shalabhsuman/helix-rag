import os
import time

from loguru import logger
from openai import OpenAI

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
BATCH_SIZE = 100  # OpenAI recommends batches of up to 100 for embeddings


def get_client() -> OpenAI:
    if os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.openai import OpenAI as LangfuseOpenAI  # type: ignore[import]
            return LangfuseOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        except ImportError:
            logger.warning("langfuse not installed. Run: uv pip install langfuse")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns one vector per text.

    Sends in batches to respect OpenAI rate limits.
    Retries once on rate limit errors with a 60-second wait.
    """
    client = get_client()
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = -(-len(texts) // BATCH_SIZE)
        logger.info(f"Embedding batch {batch_num} of {total_batches} ({len(batch)} texts)")

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBEDDING_DIMENSIONS,
            )
        except Exception as e:
            if "rate_limit" in str(e).lower():
                logger.warning("Rate limit hit. Waiting 60 seconds before retry.")
                time.sleep(60)
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    dimensions=EMBEDDING_DIMENSIONS,
                )
            else:
                raise

        all_vectors.extend([item.embedding for item in response.data])

    return all_vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string. Used at retrieval time."""
    return embed_texts([text])[0]
