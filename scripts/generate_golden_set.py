"""Generate a draft golden evaluation set from indexed PDFs using RAGAS TestsetGenerator.

This script runs once. It produces data/golden_set.json which you MUST review and
edit before running evaluation. The generator uses GPT-4o to write questions and
reference answers from your papers — treat the output as a draft, not ground truth.

Usage:
  python scripts/generate_golden_set.py
  python scripts/generate_golden_set.py --size 15
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document  # noqa: E402
from loguru import logger  # noqa: E402
from openai import OpenAI  # noqa: E402
from ragas.embeddings import embedding_factory  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.testset import TestsetGenerator  # noqa: E402

from src.ingestion.parser import parse_pdf  # noqa: E402

OUTPUT_PATH = Path("data/golden_set.json")
RAW_DIR = Path("data/raw")


def load_langchain_docs() -> list[Document]:
    docs = []
    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        logger.info(f"Parsing {pdf_path.name}")
        parsed = parse_pdf(pdf_path)
        docs.append(
            Document(
                page_content=parsed.text,
                metadata={"source": pdf_path.name, "doc_id": parsed.doc_id},
            )
        )
    logger.info(f"Loaded {len(docs)} documents")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden evaluation set")
    parser.add_argument("--size", type=int, default=10, help="Number of test samples")
    args = parser.parse_args()

    docs = load_langchain_docs()

    client = OpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    logger.info(f"Generating {args.size} test samples (this may take a few minutes)...")
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=args.size,
        raise_exceptions=False,
    )

    df = testset.to_pandas()

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "user_input": str(row.get("user_input", "")),
                "reference": str(row.get("reference", "")),
                "reference_contexts": (
                    row["reference_contexts"]
                    if isinstance(row.get("reference_contexts"), list)
                    else []
                ),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(rows, indent=2))

    logger.info(f"Saved {len(rows)} samples to {OUTPUT_PATH}")
    logger.warning(
        "IMPORTANT: Review and edit data/golden_set.json before running evaluation. "
        "Fix any questions that are vague, incorrect, or unanswerable from the papers."
    )


if __name__ == "__main__":
    main()
