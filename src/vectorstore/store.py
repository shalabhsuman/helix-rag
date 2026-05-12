import os
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.models import ChildChunk, ParentChunk

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "helix_rag")
VECTOR_SIZE = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


def ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it does not already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{COLLECTION_NAME}'")
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists")


def upsert_chunks(
    children: list[ChildChunk],
    parents: list[ParentChunk],
    vectors: list[list[float]],
) -> None:
    """Store child chunk vectors in Qdrant.

    Each point stores:
    - the vector (embedding of the child text)
    - payload with child text, parent text, doc_id, source_file

    Storing parent_text in the payload means retrieval returns everything
    the LLM needs in one call without a second lookup.
    """
    client = get_client()
    ensure_collection(client)

    parent_map = {p.chunk_id: p.text for p in parents}

    points = []
    for i, (child, vector) in enumerate(zip(children, vectors)):
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "chunk_id": child.chunk_id,
                    "parent_chunk_id": child.parent_chunk_id,
                    "parent_text": parent_map.get(child.parent_chunk_id, ""),
                    "child_text": child.text,
                    "doc_id": child.doc_id,
                    "source_file": child.source_file,
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.success(f"Stored {len(points)} vectors in Qdrant collection '{COLLECTION_NAME}'")


def search(query_vector: list[float], top_k: int = 20) -> list[dict[str, Any]]:
    """Return the top_k most similar chunks for a query vector."""
    client = get_client()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [{"score": r.score, **r.payload} for r in response.points]


def delete_by_doc_id(doc_id: str) -> None:
    """Remove all vectors belonging to a specific document."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = get_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )
    logger.success(f"Deleted all vectors for doc_id='{doc_id}'")


def collection_stats() -> dict[str, Any]:
    """Return basic stats about the collection."""
    client = get_client()
    info = client.get_collection(COLLECTION_NAME)
    return {
        "vector_count": info.points_count,
        "collection": COLLECTION_NAME,
    }
