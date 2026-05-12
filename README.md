# helix-rag

helix-rag is a production-grade agentic RAG pipeline for research-intensive domains. Point it at any collection of PDFs and it delivers grounded, cited answers through a conversational interface.

The system goes beyond basic similarity search: it combines keyword and semantic retrieval, applies a reranker before generation, enforces citation grounding through guardrails, and exposes the pipeline as a callable tool inside an agent. Every component is swappable through configuration.

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

```bash
pytest tests/ -v
```

All external calls (OpenAI, Qdrant) are mocked. Tests run in under 2 minutes with no API cost.

---

## CI

| Workflow | Trigger | What runs |
|---|---|---|
| `ci.yml` | Every push and PR | Lint (ruff), type check (mypy), unit tests |
| `eval.yml` | Merge to main | Full RAGAS and DeepEval evaluation. Fails if faithfulness < 0.80 |

Add `OPENAI_API_KEY` as a repository secret: Settings > Secrets and variables > Actions.

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
