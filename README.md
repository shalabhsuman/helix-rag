# helix-rag

End-to-end RAG pipeline for biomedical research papers. Ask questions about ecDNA research and get grounded, cited answers.

**Stack:** PyMuPDF, Qdrant, OpenAI, RAGAS, DeepEval, OpenAI Agents SDK, Gradio

---

## How it works

RAG has two phases:

**Build time** (run once, or when you add new papers):
```
PDFs -> parse text -> split into chunks -> embed chunks -> store in Qdrant
```

**Query time** (runs on every user question):
```
Question -> embed question -> retrieve top chunks -> rerank -> GPT-4o -> answer + sources
```

---

## Architecture

```
helix-rag/
  data/
    raw/              <- Put your PDFs here (gitignored)
    processed/        <- Parsed text output (gitignored)
  src/
    ingestion/        <- Phase 1: PDF parsing
    chunking/         <- Phase 2: Text splitting
    embedding/        <- Phase 3: Embedding model wrapper
    vectorstore/      <- Phase 4: Qdrant client wrapper
    retrieval/        <- Phase 5: Hybrid search + reranker
    generation/       <- Phase 6: GPT-4o prompt + call
    agent/            <- Phase 7: OpenAI Agents SDK wrapper
    ui/               <- Phase 8: Gradio chat interface
  evals/              <- RAGAS + DeepEval evaluation scripts
  scripts/            <- CLI for managing the index
  tests/              <- Unit tests (no LLM calls)
  .github/workflows/  <- CI: lint, tests, eval gate
```

---

## Tech stack decisions

### PDF Parsing: PyMuPDF

| Choice | Why | Alternative |
|---|---|---|
| PyMuPDF | Handles 2-column scientific paper layout without merging lines. Fast. | `pdfplumber` (better for tables but slower) |

### Chunking: Recursive + Parent-Child

**What:** Split each paper into small chunks for indexing, but store their full parent section for context.

**Why:** Small chunks (300 tokens) score better in similarity search. But 300 tokens is too little context for a good answer. Parent-child solves this: retrieve the small chunk, send the full parent to the LLM.

```
Full section (1200 tokens) = parent
  |- Chunk A (300 tokens)  <- indexed in Qdrant
  |- Chunk B (300 tokens)  <- indexed in Qdrant
  |- Chunk C (300 tokens)  <- indexed in Qdrant
  |- Chunk D (300 tokens)  <- indexed in Qdrant

User asks question -> matches Chunk B -> LLM gets full parent (1200 tokens)
```

| Choice | Why | Alternative |
|---|---|---|
| Recursive character splitter | Respects paragraph boundaries. Splits on `\n\n` first, then `\n`, then `.` | Fixed-size (ignores sentence boundaries) |
| Parent-child | Better answer quality without losing retrieval precision | Semantic chunking (smarter but expensive) |

### Embedding: text-embedding-3-small

| Choice | Why | Alternative |
|---|---|---|
| `text-embedding-3-small` | Strong quality, $0.02/1M tokens, easy to swap | `BAAI/bge-large-en-v1.5` (free, local, slightly stronger) |

To swap the embedding model, change `EMBEDDING_MODEL` in `.env`. **Important:** changing the model requires a full reindex because vector dimensions change.

### Vector Database: Qdrant

| Choice | Why | Alternative |
|---|---|---|
| Qdrant | Production-grade, runs locally in Docker, supports native hybrid search (BM25 + vector in one call), deploys to Qdrant Cloud with the same API | Pinecone (more name recognition, but cloud-only and requires account) |

The vector store layer is abstracted in `src/vectorstore/`. Swapping to Pinecone or Weaviate requires changing only that module.

### Retrieval: Hybrid Search + Cross-Encoder Reranker

Two-stage process:

**Stage 1: Hybrid search (fast, broad)**
- Dense vector search finds semantically similar chunks.
- BM25 keyword search finds exact terms (gene names, acronyms like ecDNA, CCND1).
- Scores are combined. Returns top 20 candidates.

**Stage 2: Cross-encoder reranker (slower, precise)**
- Takes the question AND each of the 20 chunks together.
- Scores each (question, chunk) pair as a unit.
- Returns top 5 for the LLM.

```
Question -> Hybrid search -> top 20 -> Reranker -> top 5 -> GPT-4o
```

| Choice | Why | Alternative |
|---|---|---|
| Hybrid search | Scientific papers mix exact terminology with semantic meaning | Dense-only (misses exact gene names) |
| Cross-encoder reranker | More accurate than vector similarity alone | No reranker (simpler, lower quality) |

### Generation: GPT-4o

The prompt enforces grounding. The LLM is instructed to answer only from the retrieved chunks, not from its training data. If the answer is not in the chunks, it says "I don't know."

| Choice | Why | Alternative |
|---|---|---|
| GPT-4o | Best reasoning quality, unified with Agents SDK | Claude Sonnet 3.7 (comparable quality) |

### Evaluation: RAGAS + DeepEval

| Tool | What it measures | When it runs |
|---|---|---|
| RAGAS | Faithfulness, context recall, context precision, answer relevancy | On merge to main |
| DeepEval | Same metrics + hallucination detection, runs as pytest tests | On merge to main, CI gate |

**Faithfulness** = does the answer stick to what the chunks say?
**Context recall** = did we retrieve the chunks that actually contain the answer?
**Context precision** = are the retrieved chunks relevant, or is there noise?

Quality gate: DeepEval faithfulness must be >= 0.80 for CI to pass.

### Agent: OpenAI Agents SDK

The RAG pipeline is wrapped as a tool the agent can call. This allows adding more tools later (paper summarizer, citation formatter) without rewriting the core pipeline.

```
Agent
  |- Tool: search_papers(query)  -> calls RAG pipeline, returns answer + sources
  |- Tool: list_papers()         -> returns titles of all indexed papers
```

| Choice | Why | Alternative |
|---|---|---|
| OpenAI Agents SDK | Current standard, clean API, well known in interviews | LangChain agents (heavier, more abstraction) |

### Guardrails: Guardrails AI

Two layers:

**Input (before retrieval):**
- Topic filter: rejects questions unrelated to the indexed papers.
- Prompt injection detection: blocks "ignore your instructions" attacks.

**Output (after generation):**
- Citation enforcement: if no chunks were retrieved, returns "I don't know" instead of hallucinating.

| Choice | Why | Alternative |
|---|---|---|
| Guardrails AI | Lightweight Python validators, integrates with any LLM call | NVIDIA NeMo Guardrails (more powerful, enterprise-grade, steeper learning curve) |

### UI: Gradio

| Choice | Why | Alternative |
|---|---|---|
| Gradio | 20 lines of Python, built-in chat interface, standard for ML demos | React + FastAPI (full-stack, more code) |

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for running Qdrant locally)
- An OpenAI API key

### 1. Clone the repo

```bash
git clone https://github.com/shalabhsuman/helix-rag.git
cd helix-rag
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Open .env and fill in your OPENAI_API_KEY
```

### 4. Start Qdrant locally

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

This starts Qdrant on `http://localhost:6333`. Your vector data is saved in `qdrant_storage/` so it persists between restarts.

---

## Phase 1: Ingest your papers

**What:** Parse PDFs, split into chunks, embed, and store in Qdrant.

**Why this step exists:** Before you can answer questions, you need to build the index. This is the "build time" phase described above.

### Add your PDFs

Drop all PDF files into:
```
data/raw/
```

### Run ingestion

```bash
python scripts/ingest.py --mode add
```

This will:
1. Parse every PDF in `data/raw/`
2. Split each into parent sections and child chunks
3. Embed each chunk using `text-embedding-3-small`
4. Store in Qdrant with metadata (filename, section, chunk index)

Expected output:
```
Parsing: chang_2023_ecdna.pdf ... 42 pages, 18 sections
Chunking: 18 parent sections -> 87 child chunks
Embedding: 87 chunks ... done
Stored: 87 vectors in Qdrant collection 'helix_rag'
```

---

## Phase 2: Query

**What:** Ask a question and get a grounded answer with source citations.

**Why this step exists:** This is the "query time" phase. Your question is embedded, matched against the index, and the best chunks are sent to GPT-4o.

```bash
python scripts/query.py --question "What is the relationship between ecDNA and oncogene amplification?"
```

---

## Updating the index

### Add new papers

Drop new PDFs into `data/raw/` then:

```bash
python scripts/ingest.py --mode add --input data/raw/new_paper.pdf
```

Only the new file is parsed and embedded. Existing vectors are not touched.

### Remove a paper

```bash
python scripts/ingest.py --mode delete --doc_id chang_2023_ecdna
```

All vectors tagged with that `doc_id` are deleted from Qdrant.

### Full reindex

Use this when you change the chunking strategy or embedding model. All vectors are deleted and rebuilt from scratch.

```bash
python scripts/ingest.py --mode reindex
```

**Warning:** This re-embeds every paper. It will cost a small amount in OpenAI API credits.

---

## Phase 3: Evaluation

**What:** Score the pipeline on a golden dataset of (question, expected answer) pairs.

**Why this step exists:** You need to know if the pipeline is actually good. Eyeballing answers is not reliable. RAGAS and DeepEval give you objective scores.

### Generate a golden dataset (one-time setup)

```bash
python evals/generate_golden_dataset.py
```

This uses RAGAS TestsetGenerator to create synthetic Q&A pairs from your papers. Review the output in `evals/golden_dataset.json` and delete or fix any bad pairs.

### Run evaluation

```bash
# RAGAS metrics: faithfulness, context recall, context precision
python evals/run_ragas.py

# DeepEval metrics: same + CI-compatible output
python evals/run_deepeval.py
```

---

## Phase 4: Run the agent

```bash
python src/agent/agent.py
```

---

## Phase 5: Run the UI

```bash
python src/ui/app.py
```

Opens a Gradio chat interface at `http://localhost:7860`.

---

## CI/CD

GitHub Actions runs automatically on every push and on merges to main.

| Workflow | Trigger | What runs |
|---|---|---|
| `ci.yml` | Every push and PR | Lint (ruff), type check (mypy), unit tests (pytest) |
| `eval.yml` | Merge to main | Full RAGAS + DeepEval eval suite. Fails if faithfulness < 0.80 |

Unit tests mock all LLM and Qdrant calls. They run in under 2 minutes and cost nothing.

Eval tests hit real APIs. They run only on main to avoid burning API credits on every PR.

### Required GitHub secrets

In your repo: Settings > Secrets and variables > Actions > New repository secret

| Secret | Value |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI key |
| `QDRANT_URL` | `http://localhost:6333` for local, or your Qdrant Cloud URL |

---

## Running tests locally

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## License

MIT
