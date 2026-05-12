# helix-rag

helix-rag is a production-grade agentic RAG pipeline for research-intensive domains. Point it at any collection of PDFs and it delivers grounded, cited answers through a conversational interface.

The system goes beyond basic similarity search: it combines keyword and semantic retrieval, applies a reranker before generation, enforces citation grounding through guardrails, and exposes the pipeline as a callable tool inside an agent. Every component is swappable through configuration.

---

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Design decisions](#design-decisions)
- [Monitoring dashboards](#monitoring-dashboards)
- [What indexing actually means](#what-indexing-actually-means)
- [What a vector database actually looks like](#what-a-vector-database-actually-looks-like)
- [Build phases](#build-phases)
- [Setup](#setup)
- [Index management](#index-management)
- [Querying](#querying)
- [Evaluation](#evaluation)
- [Running the agent](#running-the-agent)
- [Tests](#tests)
- [CI](#ci)
- [Architecture](#architecture)
- [License](#license)

---

## Features

- Hybrid search (BM25 + dense vector) so exact terminology is never missed
- Parent-child chunking for precise retrieval with rich context at generation time
- Cross-encoder reranking before the LLM call, not just raw vector similarity
- Guardrails on both input and output to prevent hallucination and prompt injection
- Agent layer built on OpenAI Agents SDK so the RAG pipeline is one tool among many
- Automated evaluation with RAGAS and DeepEval, gated in CI

---

## How it works

The pipeline has two phases.

**Indexing** (runs once, or when documents are updated):

```
PDFs -> text extraction -> chunking -> embedding -> Qdrant
```

**Querying** (runs on every request):

```
Question -> embed -> hybrid search -> rerank -> GPT-4o -> answer + citations
```

Documents are embedded once and stored. Querying is fast and cheap after the index is built.

---

## Project structure

```
helix-rag/
  data/
    raw/              <- Drop your PDFs here (not tracked by git)
    processed/        <- Intermediate parsed output (not tracked by git)
  src/
    ingestion/        <- PDF text extraction
    chunking/         <- Parent-child text splitting
    embedding/        <- Embedding model wrapper (swappable)
    vectorstore/      <- Qdrant client wrapper (swappable)
    retrieval/        <- Hybrid search + cross-encoder reranker
    generation/       <- Prompt construction + GPT-4o call
    agent/            <- OpenAI Agents SDK: RAG as a callable tool
    ui/               <- Gradio chat interface
  evals/              <- RAGAS and DeepEval evaluation scripts
  scripts/            <- CLI for index management and querying
  tests/              <- Unit tests with mocked external calls
  .github/workflows/  <- CI: lint, tests, and eval quality gate
```

---

## Design decisions

### PDF parsing: PyMuPDF

Scientific papers are typically two-column PDFs. Most parsers concatenate the columns into broken text. PyMuPDF preserves layout and handles multi-column documents correctly. It is also significantly faster than alternatives for batch ingestion.

The alternative is `pdfplumber`, which is more accurate on complex tables. For document sets with critical tabular data, it can be swapped in at `src/ingestion/`.

### Chunking: recursive splitting with parent-child storage

Small chunks produce better similarity scores because they are focused. But small chunks often lack enough context for a quality answer. Parent-child chunking separates these concerns:

```
Parent section (1200 tokens) - stored separately, sent to LLM

  Child A (300 tokens)  <- indexed and searched
  Child B (300 tokens)  <- indexed and searched
  Child C (300 tokens)  <- indexed and searched
  Child D (300 tokens)  <- indexed and searched
```

Retrieval targets child chunks. Generation receives the full parent. Precision and context are both preserved.

The alternative is semantic chunking, which splits on topic boundaries detected by embedding similarity. It is more accurate but slower and more expensive to run at ingestion time.

### Embedding: text-embedding-3-small

Strong retrieval quality at low cost. Supports Matryoshka representation, meaning you can reduce vector dimensions at query time to trade a small amount of accuracy for speed.

To switch to a local model (`BAAI/bge-large-en-v1.5`), change `EMBEDDING_MODEL` in `.env`. A full reindex is required when changing models because the vector dimensions change.

### Vector database: Qdrant

Qdrant supports native hybrid search (dense + sparse in a single query), runs on-premise or on Qdrant Cloud with the same API, and has strong filtering on document metadata. The vector store is fully abstracted at `src/vectorstore/` and can be replaced with Pinecone, Weaviate, or pgvector without touching the rest of the pipeline.

### Retrieval: hybrid search with reranking

The retrieval stage runs in two steps:

1. Hybrid search combines dense vector similarity with BM25 keyword scoring. This handles both semantic queries and exact-match lookups. Returns the top 20 candidates.
2. A cross-encoder reranker scores each (question, chunk) pair jointly rather than independently. The top 5 candidates are passed to the LLM.

```
Question -> hybrid search (top 20) -> cross-encoder reranker (top 5) -> GPT-4o
```

Dense-only retrieval is simpler but misses exact terminology that is common in technical and scientific domains.

### Generation: GPT-4o with grounding constraint

The system prompt constrains the LLM to answer only from retrieved context. If the context does not contain sufficient information, the response is "I do not have enough information to answer that." This constraint is enforced at the prompt level and validated by the output guardrail.

### Guardrails

Input guardrails run before retrieval:

- Topic relevance check: rejects questions outside the scope of indexed documents
- Prompt injection detection: blocks attempts to override system instructions

Output guardrails run after generation:

- Citation check: if no relevant chunks were retrieved, the response is blocked before it reaches the user
- Faithfulness check: flags responses that contradict the retrieved context

### Evaluation: RAGAS and DeepEval

Three metrics are tracked against a golden dataset of 30 to 40 hand-reviewed question and answer pairs:

| Metric | What it measures |
|---|---|
| Faithfulness | Does the answer reflect what the retrieved chunks say? |
| Context recall | Did retrieval surface the chunks that actually contain the answer? |
| Context precision | Are the retrieved chunks relevant, or is there noise? |

DeepEval runs the evaluation as pytest tests. The CI pipeline blocks merges to main if faithfulness falls below 0.80.

### Agent: OpenAI Agents SDK

The RAG pipeline is registered as a tool inside an OpenAI agent. The agent decides when to invoke it based on the user's question. Additional tools (summarization, citation formatting, cross-document comparison) can be added without modifying the retrieval or generation pipeline.

### UI: Gradio

Gradio provides a chat interface with source citation rendering. It runs as a standalone process and connects to the pipeline over a clean internal interface, making it straightforward to replace with a different frontend.

---

## Monitoring dashboards

Every tool in this stack has a way to inspect what is happening. Bookmark these.

| Tool | Local dashboard | Online dashboard | What to check |
|---|---|---|---|
| **Qdrant** | http://localhost:6333/dashboard | [Qdrant Cloud](https://cloud.qdrant.io) (if deployed) | Browse vectors, run test queries, check collection stats |
| **OpenAI** | n/a | [platform.openai.com/usage](https://platform.openai.com/usage) | API token usage, cost per day, which models were called |
| **GitHub Actions** | n/a | `github.com/shalabhsuman/helix-rag/actions` | CI run history, lint/test pass or fail, eval scores |
| **Gradio UI** (Phase 7) | http://localhost:7860 | n/a | The chat interface itself |

To open the Qdrant dashboard, Qdrant must be running in Docker first (`docker ps` to check). Then open http://localhost:6333/dashboard in your browser. Click the `helix_rag` collection, then click **Points** to browse all 821 stored chunks.

---

## What indexing actually means

The word "indexing" gets used in two ways in this project. They are related but different.

---

**Meaning 1: the ingestion pipeline (our code)**

When you run `python scripts/ingest.py --mode add`, that is what most people in this project call "indexing." It means: take the PDFs, parse them, split them into chunks, embed each chunk, and store the vectors in Qdrant. This is a one-time operation per document. The result is 821 stored vectors.

---

**Meaning 2: the internal data structure Qdrant builds (automatic)**

Once vectors are stored, Qdrant needs a way to find similar ones quickly at query time. The naive approach would be: take the query vector and compare it against all 821 stored vectors one by one. For 821 vectors that is fine. For 10 million vectors it would take seconds per query.

To make search fast at scale, Qdrant builds an internal index called **HNSW** (Hierarchical Navigable Small World). You do not build this manually. Qdrant builds it automatically as you insert vectors.

**How HNSW works (simplified):**

Think of it like a highway system. Instead of checking every city to find the nearest one, you:

1. Start at a high level (like interstate highways) and find roughly the right region
2. Drop down to a mid level (state roads) and narrow further
3. Drop to the lowest level (local streets) and find the exact nearest point

```
Layer 2 (sparse):   A -------- F -------- K
                    |                     |
Layer 1 (medium):   A --- C --- F --- H -- K
                    |    |     |    |
Layer 0 (dense):    A-B-C-D-E-F-G-H-I-J-K   <- all 821 chunks live here
```

A query starts at the top layer and navigates down. Instead of checking all 821 points, it checks maybe 30 to 50 on the way down and finds the nearest match. This is why vector search is fast even at scale.

**The practical tradeoff:**

| | Brute force (no index) | HNSW index |
|---|---|---|
| Build time | None | A few seconds for 821 vectors |
| Query speed | Slow at scale | Fast even at millions |
| Accuracy | Exact | Approximate (but very close) |
| Memory | Low | Higher (graph structure stored in RAM) |

For 821 vectors, you will not notice the difference. But the index is what makes Qdrant production-grade for larger collections.

**How to see the index status in Qdrant:**

Open http://localhost:6333/dashboard, click `helix_rag`, and look at the **Status** field. When it says `green`, the index is built and ready. When vectors are being inserted, it briefly shows `yellow`.

---

## What a vector database actually looks like

A vector database is not a spreadsheet and not a relational table. Each stored item is a JSON object with two parts: a vector and a payload.

The **vector** is a list of numbers (1536 numbers in this project) that encodes the meaning of the text mathematically. You cannot read it as a human. Qdrant uses it to find chunks that are semantically similar to a question.

The **payload** is regular metadata attached to that vector. It works like columns in a normal database.

Here is what one stored point actually looks like:

```json
{
  "id": 1,
  "vector": [0.041, 0.034, 0.064, 0.045, ...],
  "payload": {
    "doc_id": "bailey_2024_ecdna_origins_impact",
    "source_file": "bailey_2024_ecdna_origins_impact.pdf",
    "chunk_id": "bailey_2024_ecdna_origins_impact_child_1",
    "parent_chunk_id": "bailey_2024_ecdna_origins_impact_parent_0",
    "child_text": "17.1% of tumour samples contain ecDNA...",
    "parent_text": "Origins and impact of extrachromosomal DNA. Chris Bailey..."
  }
}
```

If you think of it as a spreadsheet, it looks like this:

| id | vector | doc_id | chunk_id | parent_chunk_id | child_text | parent_text |
|---|---|---|---|---|---|---|
| 0 | [0.070, 0.027, ...] | bailey_2024... | ..._child_0 | ..._parent_0 | "Origins and impact..." | "Origins and impact..." |
| 1 | [0.041, 0.034, ...] | bailey_2024... | ..._child_1 | ..._parent_0 | "17.1% of tumour samples..." | "Origins and impact..." |
| 2 | [0.019, 0.051, ...] | bailey_2024... | ..._child_2 | ..._parent_1 | "ecDNA amplification..." | "ecDNA is a driver..." |

There are 821 rows in this project. One row per child chunk. Chunks from the same parent section share the same `parent_chunk_id`. When retrieval finds a child chunk, it uses the stored `parent_text` to send richer context to the LLM.

---

## Build phases

The pipeline is built in phases. Each phase is independently testable.

| Phase | What it builds | Status |
|---|---|---|
| 1 | PDF parsing and parent-child chunking | Done |
| 2 | Embedding and Qdrant vector storage | Done |
| 3 | Hybrid search and cross-encoder reranking | In progress |
| 4 | GPT-4o generation with grounding constraint | Pending |
| 5 | RAGAS and DeepEval evaluation pipeline | Pending |
| 6 | OpenAI Agents SDK wrapper | Pending |
| 7 | Gradio chat UI | Pending |

---

## Setup

Requirements: Python 3.11+, Docker, OpenAI API key.

### 1. Clone the repo

```bash
git clone https://github.com/shalabhsuman/helix-rag.git
cd helix-rag
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. All other defaults work for local development.

### 4. Start Qdrant

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Data is persisted in `qdrant_storage/` and survives container restarts. Keep this running in a separate terminal.

---

## Index management

### Build the index

Drop PDFs into `data/raw/`, then run:

```bash
python scripts/ingest.py --mode add
```

### Add documents

```bash
python scripts/ingest.py --mode add --input data/raw/new_paper.pdf
```

### Remove a document

```bash
python scripts/ingest.py --mode delete --doc_id author_year_title
```

### Rebuild the full index

Required when changing the embedding model or chunking strategy. Existing vectors are incompatible after either change.

```bash
python scripts/ingest.py --mode reindex
```

---

## Querying

```bash
python scripts/query.py --question "Your question here"
```

Or start the chat UI:

```bash
python src/ui/app.py
```

Available at `http://localhost:7860`.

---

## Evaluation

### Generate the golden dataset (one-time)

```bash
python evals/generate_golden_dataset.py
```

Produces synthetic question and answer pairs from indexed documents using RAGAS TestsetGenerator. Review and edit `evals/golden_dataset.json` before running evaluation.

### Run evaluation

```bash
python evals/run_ragas.py
python evals/run_deepeval.py
```

---

## Running the agent

```bash
python src/agent/agent.py
```

---

## Tests

All tests live in `tests/`. Every external call (OpenAI, Qdrant) is mocked, so tests run in under 30 seconds with no API cost.

```bash
pytest tests/ -v
```

### What each test file covers

| File | Tests | What it verifies |
|---|---|---|
| `test_ingestion.py` | 5 | PDF parser cleans hyphenated line breaks, collapses whitespace, returns correct doc_id |
| `test_chunking.py` | 7 | Parent-child splitting produces correct sizes, every child references a valid parent, all chunk IDs are unique |
| `test_embedding.py` | 4 | Embedder returns one vector per text, correct dimensions, batches large inputs correctly |
| `test_vectorstore.py` | 4 | Qdrant upsert is called with correct points, payload contains parent text, search and delete work |
| `test_retrieval.py` | 7 | BM25 respects top_k, RRF boosts chunks appearing in both lists, RRF formula is mathematically correct, reranker sorts by score |

### When tests run

| Trigger | What runs | Cost |
|---|---|---|
| Every push to any branch | All 27 unit tests | Free (no API calls) |
| Every pull request | All 27 unit tests | Free |
| Merge to main | All 27 unit tests + full eval suite | Small API cost |
| Manual trigger (eval workflow) | Full RAGAS + DeepEval against golden dataset | API cost |

### The three types of checks in CI

These are different things. Only pytest requires you to write code.

| Tool | You write it? | What it does | How to run locally |
|---|---|---|---|
| `pytest` | Yes. You write test functions. | Runs your code and checks the results are correct | `pytest tests/ -v` |
| `ruff` | No. Just configured in `pyproject.toml`. | Checks code style: unused imports, lines too long, formatting | `ruff check src/ tests/` |
| `mypy` | No. Just configured in `pyproject.toml`. | Checks type hints: catches passing wrong types to functions | `mypy src/` |

If any of the three fails, the CI run is marked as failed. To run all three locally before pushing:

```bash
ruff check src/ tests/ && mypy src/ && pytest tests/ -v
```

---

## CI

Two workflows live in `.github/workflows/`. GitHub runs them automatically.

| Workflow | File | Trigger | What runs | Time |
|---|---|---|---|---|
| CI | `ci.yml` | Every push and PR | Lint (ruff) + type check (mypy) + unit tests (pytest) | ~2 min |
| Eval | `eval.yml` | Manual only (until Phase 5) | Full RAGAS + DeepEval evaluation against golden dataset | ~10 min |

The eval workflow is set to manual-only until the evaluation pipeline (Phase 5) is built. After Phase 5, it will run automatically on every merge to main and block the merge if faithfulness drops below 0.80.

To add your API key as a GitHub secret (required for the eval workflow): go to your repo on GitHub, then Settings > Secrets and variables > Actions > New repository secret. Add `OPENAI_API_KEY`.

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingest["Indexing Pipeline (build time)"]
        A[PDF Files] --> B[PyMuPDF\nText Extraction]
        B --> C[Recursive Splitter\nParent-Child Chunks]
        C --> D[OpenAI Embeddings\ntext-embedding-3-small]
        D --> E[(Qdrant\nVector Store)]
    end

    subgraph Query["Query Pipeline (request time)"]
        F[User Question] --> G[Input Guardrails\ntopic filter + injection check]
        G --> H[OpenAI Embeddings]
        H --> I[Hybrid Search\nDense + BM25]
        E --> I
        I --> J[Cross-Encoder\nReranker]
        J --> K[GPT-4o\ngrounded generation]
        K --> L[Output Guardrails\ncitation check]
        L --> M[Answer + Sources]
    end

    subgraph AgentLayer["Agent Layer"]
        N[OpenAI Agent] -->|invokes tool| F
        M --> N
    end

    subgraph Eval["Evaluation"]
        O[Golden Dataset] --> P[RAGAS + DeepEval]
        M --> P
        P --> Q{Faithfulness\n>= 0.80?}
        Q -->|pass| R[Merge to main]
        Q -->|fail| S[Block merge]
    end
```

---

## License

MIT
