"""Test retrieval from the command line.

Usage:
  python scripts/query.py --question "What is the relationship between ecDNA and oncogene amplification?"
  python scripts/query.py --question "How does ecDNA inherit during cell division?" --top_k 3
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the helix-rag retrieval pipeline")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to return")
    args = parser.parse_args()

    retriever = Retriever()
    results = retriever.retrieve(args.question, top_k=args.top_k)

    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    print(f"{'='*60}\n")

    for i, chunk in enumerate(results, 1):
        print(f"--- Result {i} | score={chunk.score:.4f} ---")
        print(f"Source : {chunk.source_file}")
        print(f"Chunk  : {chunk.chunk_id}")
        print(f"Text   : {chunk.child_text[:400]}")
        print()


if __name__ == "__main__":
    main()
