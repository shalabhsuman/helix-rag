"""CLI for managing the helix-rag document index.

Usage:
  python scripts/ingest.py --mode add                           # index all PDFs in data/raw/
  python scripts/ingest.py --mode add --input data/raw/paper.pdf  # index one PDF
  python scripts/ingest.py --mode delete --doc_id kim_2020_...  # remove a paper (Phase 2)
  python scripts/ingest.py --mode reindex                       # rebuild entire index (Phase 2)
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.splitter import split_document
from src.ingestion.parser import parse_pdf

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def add(input_path: Path | None = None) -> None:
    pdfs = [input_path] if input_path else sorted(RAW_DIR.glob("*.pdf"))

    if not pdfs:
        logger.warning(f"No PDFs found in {RAW_DIR}. Drop your papers there and try again.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    total_parents = 0
    total_children = 0

    for pdf in pdfs:
        logger.info(f"--- Processing {pdf.name} ---")
        doc = parse_pdf(pdf)
        chunks = split_document(doc)

        output = {
            "doc_id": doc.doc_id,
            "source_file": doc.source_file,
            "page_count": doc.page_count,
            "parent_count": len(chunks.parents),
            "child_count": len(chunks.children),
            "parents": [p.model_dump() for p in chunks.parents],
            "children": [c.model_dump() for c in chunks.children],
        }

        out_path = PROCESSED_DIR / f"{doc.doc_id}.json"
        out_path.write_text(json.dumps(output, indent=2))

        total_parents += len(chunks.parents)
        total_children += len(chunks.children)
        logger.success(
            f"Saved {out_path.name}: {len(chunks.parents)} parents, {len(chunks.children)} children"
        )

    logger.success(
        f"\nDone. {len(pdfs)} papers -> {total_parents} parent chunks, {total_children} child chunks."
    )
    logger.info("Inspect the output in data/processed/ before running Phase 2 (embedding + Qdrant).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the helix-rag document index")
    parser.add_argument("--mode", choices=["add", "delete", "reindex"], required=True)
    parser.add_argument("--input", type=Path, help="Single PDF to ingest (use with --mode add)")
    parser.add_argument("--doc_id", help="Document ID to remove (use with --mode delete)")
    args = parser.parse_args()

    if args.mode == "add":
        add(input_path=args.input)
    elif args.mode == "delete":
        logger.warning("--mode delete requires Qdrant (Phase 2). Not yet implemented.")
    elif args.mode == "reindex":
        logger.warning("--mode reindex requires Qdrant (Phase 2). Running add for now.")
        add()


if __name__ == "__main__":
    main()
