"""Run RAGAS evaluation against the golden set.

Loads data/golden_set.json, runs each question through the full RAG pipeline,
scores the results with RAGAS, prints a score table, and saves results to
data/eval_results.json for the CI threshold tests to read.

Usage:
  python scripts/run_eval.py
"""

import json
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from loguru import logger  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics._answer_relevance import answer_relevancy  # noqa: E402
from ragas.metrics._context_precision import context_precision  # noqa: E402
from ragas.metrics._context_recall import context_recall  # noqa: E402
from ragas.metrics._faithfulness import faithfulness  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from src.generation.generator import generate  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402

GOLDEN_SET_PATH = Path("data/golden_set.json")
RESULTS_PATH = Path("data/eval_results.json")


def main() -> None:
    if not GOLDEN_SET_PATH.exists():
        logger.error(
            f"{GOLDEN_SET_PATH} not found. "
            "Run scripts/generate_golden_set.py first and review the output."
        )
        sys.exit(1)

    golden = json.loads(GOLDEN_SET_PATH.read_text())
    logger.info(f"Loaded {len(golden)} golden samples")

    retriever = Retriever()

    samples = []
    for i, row in enumerate(golden):
        question = row["user_input"]
        logger.info(f"[{i+1}/{len(golden)}] {question[:80]}")

        chunks = retriever.retrieve(question, top_k=5)
        result = generate(question, chunks)

        samples.append(
            SingleTurnSample(
                user_input=question,
                reference=row["reference"],
                retrieved_contexts=[c.parent_text for c in chunks],
                response=result.answer,
            )
        )

    dataset = EvaluationDataset(samples=samples)

    # gpt-4o-mini for judging: 200k TPM on Tier 1 vs 30k for gpt-4o.
    # Accurate enough for evaluation scoring; pipeline still uses gpt-4o for generation.
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    faithfulness.llm = llm
    context_recall.llm = llm
    context_precision.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings

    metrics = [faithfulness, context_recall, context_precision, answer_relevancy]

    run_config = RunConfig(timeout=120, max_retries=3, max_wait=60)

    logger.info("Scoring with RAGAS (GPT-4o as judge)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    def _mean(values: list) -> float:
        valid = [v for v in values if v is not None and not math.isnan(float(v))]
        return round(sum(valid) / len(valid), 4) if valid else 0.0

    scores = {
        "faithfulness": _mean(results["faithfulness"]),
        "context_recall": _mean(results["context_recall"]),
        "context_precision": _mean(results["context_precision"]),
        "answer_relevancy": _mean(results["answer_relevancy"]),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(scores, indent=2))

    print(f"\n{'='*50}")
    print("RAGAS Evaluation Results")
    print(f"{'='*50}")
    print(f"  Faithfulness       {scores['faithfulness']:.4f}  (target >= 0.80)")
    print(f"  Context Recall     {scores['context_recall']:.4f}  (target >= 0.75)")
    print(f"  Context Precision  {scores['context_precision']:.4f}  (target >= 0.70)")
    print(f"  Answer Relevancy   {scores['answer_relevancy']:.4f}  (target >= 0.80)")
    print(f"{'='*50}")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
