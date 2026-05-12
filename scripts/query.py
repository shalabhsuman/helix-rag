"""Run an end-to-end query through the helix-rag pipeline.

Usage:
  python scripts/query.py --question "What is the relationship between ecDNA and oncogene amplification?"
  python scripts/query.py --question "How does ecDNA inherit during cell division?" --top_k 3
  python scripts/query.py --question "..." --chunks-only   # skip generation, show retrieved chunks only
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.generator import generate
from src.retrieval.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the helix-rag pipeline")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Show retrieved chunks without calling GPT-4o",
    )
    args = parser.parse_args()

    retriever = Retriever()
    chunks = retriever.retrieve(args.question, top_k=args.top_k)

    if args.chunks_only:
        print(f"\n{'='*60}")
        print(f"Question: {args.question}")
        print(f"{'='*60}\n")
        for i, chunk in enumerate(chunks, 1):
            print(f"--- Result {i} | score={chunk.score:.4f} ---")
            print(f"Source : {chunk.source_file}")
            print(f"Text   : {chunk.child_text[:400]}")
            print()
        return

    result = generate(args.question, chunks)

    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    print(f"{'='*60}\n")
    print(result.answer)
    print(f"\n{'─'*60}")
    print(f"Sources ({result.chunks_used} chunks):")
    for source in result.sources:
        print(f"  - {source}")


if __name__ == "__main__":
    main()
