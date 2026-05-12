"""CLI for managing the helix-rag document index.

Usage:
  python scripts/ingest.py --mode add                              # index all PDFs in data/raw/
  python scripts/ingest.py --mode add --input data/raw/paper.pdf  # index one PDF
  python scripts/ingest.py --mode delete --doc_id kim_2020_...    # remove a paper from Qdrant
  python scripts/ingest.py --mode reindex                         # rebuild entire index from scratch
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.splitter import split_document
from src.embedding.embedder import embed_texts
from src.ingestion.parser import parse_pdf
from src.models import ChildChunk, ParentChunk
from src.vectorstore.store import collection_stats, delete_by_doc_id, upsert_chunks

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def add(input_path: Path | None = None) -> None:
    pdfs = [input_path] if input_path else sorted(RAW_DIR.glob("*.pdf"))

    if not pdfs:
        logger.warning(f"No PDFs found in {RAW_DIR}. Drop your papers there and try again.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_children: list[ChildChunk] = []
    all_parents: list[ParentChunk] = []

    # Step 1: Parse and chunk all PDFs, cache results to disk
    for pdf in pdfs:
        cache_path = PROCESSED_DIR / f"{pdf.stem}.json"

        if cache_path.exists():
            logger.info(f"Using cached chunks for {pdf.name}")
            data = json.loads(cache_path.read_text())
            parents = [ParentChunk(**p) for p in data["parents"]]
            children = [ChildChunk(**c) for c in data["children"]]
        else:
            logger.info(f"--- Parsing {pdf.name} ---")
            doc = parse_pdf(pdf)
            chunks = split_document(doc)
            parents = chunks.parents
            children = chunks.children

            output = {
                "doc_id": doc.doc_id,
                "source_file": doc.source_file,
                "page_count": doc.page_count,
                "parent_count": len(parents),
                "child_count": len(children),
                "parents": [p.model_dump() for p in parents],
                "children": [c.model_dump() for c in children],
            }
            cache_path.write_text(json.dumps(output, indent=2))

        all_parents.extend(parents)
        all_children.extend(children)

    logger.info(f"Total: {len(all_children)} child chunks to embed across {len(pdfs)} papers")

    # Step 2: Embed all child chunks
    logger.info("Embedding child chunks via OpenAI...")
    child_texts = [c.text for c in all_children]
    vectors = embed_texts(child_texts)

    # Step 3: Store in Qdrant
    logger.info("Storing vectors in Qdrant...")
    upsert_chunks(all_children, all_parents, vectors)

    stats = collection_stats()
    logger.success(
        f"Done. Qdrant collection '{stats['collection']}' now contains {stats['vector_count']} vectors."
    )


def delete(doc_id: str) -> None:
    logger.info(f"Deleting vectors for doc_id='{doc_id}'")
    delete_by_doc_id(doc_id)

    cache_path = PROCESSED_DIR / f"{doc_id}.json"
    if cache_path.exists():
        cache_path.unlink()
        logger.info(f"Removed cache file {cache_path.name}")


def reindex() -> None:
    logger.warning("Reindexing: deleting all existing vectors and rebuilding from scratch.")
    from qdrant_client import QdrantClient
    from src.vectorstore.store import COLLECTION_NAME, QDRANT_URL

    client = QdrantClient(url=QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted collection '{COLLECTION_NAME}'")

    # Clear cache so PDFs are re-parsed
    for f in PROCESSED_DIR.glob("*.json"):
        f.unlink()
    logger.info("Cleared processed cache")

    add()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the helix-rag document index")
    parser.add_argument("--mode", choices=["add", "delete", "reindex"], required=True)
    parser.add_argument("--input", type=Path, help="Single PDF to ingest (use with --mode add)")
    parser.add_argument("--doc_id", help="Document ID to remove (use with --mode delete)")
    args = parser.parse_args()

    if args.mode == "add":
        add(input_path=args.input)
    elif args.mode == "delete":
        if not args.doc_id:
            logger.error("--doc_id is required for --mode delete")
            sys.exit(1)
        delete(args.doc_id)
    elif args.mode == "reindex":
        reindex()


if __name__ == "__main__":
    main()
