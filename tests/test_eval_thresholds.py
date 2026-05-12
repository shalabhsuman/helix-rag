"""CI threshold tests for RAGAS evaluation scores.

These tests load data/eval_results.json (written by scripts/run_eval.py)
and assert that every metric meets its minimum quality bar. They run in
eval.yml after run_eval.py — not in the standard unit test CI run.

If any assertion fails, the merge to main is blocked.
"""

import json
from pathlib import Path

import pytest

RESULTS_PATH = Path("data/eval_results.json")

THRESHOLDS = {
    "faithfulness": 0.75,
    "context_recall": 0.75,
    "context_precision": 0.70,
    "answer_relevancy": 0.80,
}


@pytest.fixture(scope="module")
def eval_scores() -> dict:
    if not RESULTS_PATH.exists():
        pytest.skip(
            f"{RESULTS_PATH} not found. Run scripts/run_eval.py before these tests."
        )
    return json.loads(RESULTS_PATH.read_text())


def test_faithfulness(eval_scores: dict) -> None:
    score = eval_scores["faithfulness"]
    threshold = THRESHOLDS["faithfulness"]
    assert score >= threshold, f"Faithfulness {score:.4f} is below threshold {threshold}"


def test_context_recall(eval_scores: dict) -> None:
    score = eval_scores["context_recall"]
    threshold = THRESHOLDS["context_recall"]
    assert score >= threshold, f"Context recall {score:.4f} is below threshold {threshold}"


def test_context_precision(eval_scores: dict) -> None:
    score = eval_scores["context_precision"]
    threshold = THRESHOLDS["context_precision"]
    assert score >= threshold, (
        f"Context precision {score:.4f} is below threshold {threshold}"
    )


def test_answer_relevancy(eval_scores: dict) -> None:
    score = eval_scores["answer_relevancy"]
    threshold = THRESHOLDS["answer_relevancy"]
    assert score >= threshold, (
        f"Answer relevancy {score:.4f} is below threshold {threshold}"
    )
