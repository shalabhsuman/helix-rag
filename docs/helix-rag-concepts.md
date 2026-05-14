# helix-rag: Concepts Reference

Running notes on every concept introduced during the build.
One entry per concept: what it is, why we use it, where it lives in the code.

---

## PDF Parsing

**What:** Extracting raw text from PDF files.
**Why:** PDFs are the standard format for scientific papers. We need the text before we can chunk or embed it.
**Tool:** PyMuPDF (`fitz`). Faster and more accurate on scientific PDFs than pdfplumber or PyPDF2.
**Code:** `src/ingestion/parser.py`

---

## Chunking

**What:** Splitting a long document into smaller pieces so they fit in a model's context window.
**Why:** You cannot embed a 40-page paper as one unit. Smaller chunks also give more precise retrieval.
**Strategy:** Recursive character splitting with a parent-child hierarchy.

- **Child chunks** (300 tokens): what gets embedded and searched
- **Parent chunks** (1200 tokens): what gets sent to the LLM as context

The child finds the right neighborhood. The parent provides the full story.
**Code:** `src/ingestion/chunker.py`

---

## Embeddings

**What:** Converting text into a list of numbers (a vector) that captures meaning.
**Why:** Vectors let you measure similarity mathematically. Similar sentences land close together in vector space.
**Model:** `text-embedding-3-small` (OpenAI). 1536 dimensions. Fast and cheap.
**Important:** The same model must be used at index time and query time. Mixing models breaks retrieval.
**Code:** `src/embedding/embedder.py`

---

## Vector Database

**What:** A database optimized for storing and searching vectors by similarity.
**Why:** Standard databases search by exact match. A vector database finds the nearest neighbors to a query vector.
**Tool:** Qdrant. Runs locally in Docker. Uses HNSW index for fast approximate nearest-neighbor search.
**Similarity metric:** Cosine similarity — measures the angle between two vectors, not their magnitude.
**Code:** `src/vectorstore/store.py`

---

## Hybrid Search

**What:** Combining two retrieval methods and merging their results.
**Why:** Dense (vector) search is good at semantic meaning. BM25 is good at exact keyword matches. Neither alone is best. Together they cover more ground.

- **BM25:** Classic keyword search. Scores documents by term frequency. Fast, no GPU needed.
- **Dense:** Embedding-based similarity search against Qdrant.
- **Fusion:** Results from both are merged using Reciprocal Rank Fusion (RRF).

**Code:** `src/retrieval/retriever.py`

---

## Cross-Encoder Reranking

**What:** A second pass that re-scores retrieved chunks using a more powerful model.
**Why:** The initial retrieval (BM25 + dense) uses fast approximate methods. The reranker reads the query and each chunk together and gives a more accurate relevance score.
**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally via sentence-transformers).
**Trade-off:** Slower than retrieval, but only runs on the top 20 candidates, not the full index.
**Code:** `src/retrieval/retriever.py` — `_rerank()` method

---

## RAG Generation

**What:** Using retrieved chunks as context to answer a question with a language model.
**Why:** Without retrieval, GPT answers from training memory and may hallucinate or give uncited answers. With retrieval, the answer is grounded in actual documents.
**Grounding constraint:** The system prompt tells GPT-4o to only use the provided context and always cite the source.
**Code:** `src/generation/generator.py`

---

## Input Guardrails

**What:** A check that runs before any LLM call to screen the user's input.
**Why:** Prevents off-topic, harmful, or out-of-scope queries from reaching the expensive pipeline.
**Implementation:** A fast GPT call with a yes/no classification prompt. If the query is out of scope, the pipeline stops early and returns a message without running retrieval or generation.
**Code:** `src/guardrails/guardrails.py`

---

## Agent Layer

**What:** A layer that decides which tool to call based on the user's message.
**Why:** Not every question needs retrieval. "What papers do you have?" needs a file list, not a vector search. The agent routes each question to the right tool.
**Tool:** OpenAI Agents SDK. Wraps GPT-4o with tool definitions.
**Tools exposed:**
- `search_papers` — runs the full RAG pipeline (retrieve + rerank + generate)
- `list_papers` — reads `data/raw/` and returns filenames

**How a turn works:**
1. Turn 1: LLM reads the question and tool definitions, decides which tool to call
2. Tool executes locally on your machine
3. Turn 2: LLM reads the tool output and writes the final answer

Minimum 2 LLM calls per question that uses a tool.
**Code:** `src/agent/agent.py`

---

## Conversation Memory (In-Context)

**What:** Keeping the full conversation history and sending it to the LLM on every turn.
**Why:** Allows follow-up questions. "Summarize the first one" only makes sense if the LLM remembers what "the first one" was.
**How:** `result.to_input_list()` returns the full conversation after each turn. The next call passes the full history as input.
**Limitation:** History grows by 2 messages per turn. After 20 turns, token count is substantial. There is no summarization or compression — the full transcript is sent every time.
**Scope:** In-session only. Restarting the server or closing the browser tab clears history.
**Code:** `src/ui/app.py` — `gr.State([])` + `to_input_list()`

---

## Gradio UI

**What:** A Python library for building browser-based interfaces for ML models.
**Why:** Faster to build than a React frontend. Handles sessions, state, and streaming natively.
**Key component:** `gr.State` — per-session state that persists across interactions within one browser tab without using global variables.
**Code:** `src/ui/app.py`

---

## Streaming

**What:** Sending tokens to the UI as they are generated instead of waiting for the full response.
**Why:** Users feel a 15-second pause as broken. Seeing tokens appear immediately feels responsive even if total time is the same.
**How:** `Runner.run_streamed()` returns events as the LLM writes. We listen for `response.output_text.delta` events and yield each token to Gradio.
**Code:** `src/ui/app.py` — `respond()` async generator

---

## Tracing

**What:** A record of every LLM call, tool invocation, and token count for a single request.
**Why:** Without traces you cannot debug why the agent gave a wrong answer, which tool fired, or how much a query cost.
**Tool:** OpenAI platform (`platform.openai.com/traces`). Every `Runner.run_streamed()` call creates one trace automatically.
**Trace naming:** We set `RunConfig(workflow_name=f"helix-rag | {message[:60]}")` so each trace is searchable by question.
**Code:** `src/ui/app.py` — `RunConfig`

---

## Observability (Langfuse)

**What:** A dashboard that logs every OpenAI API call with cost, latency, and token counts.
**Why:** The OpenAI platform shows traces but not dollar cost. Langfuse shows cost per request across all models including embeddings.
**Tool:** Langfuse. Open source, self-hostable, free cloud tier at `cloud.langfuse.com`.
**Activation:** Set `LANGFUSE_SECRET_KEY` in `.env`. Both the agent and embedding clients wrap automatically.
**Key metrics tracked:**

| Metric | What it means |
|---|---|
| TTFT | Time to first token — how long before the response starts |
| TTLT | Time to last token — total end-to-end time |
| Input tokens | Tokens sent to OpenAI (cost: $2.50/1M for gpt-4o) |
| Output tokens | Tokens received (cost: $10.00/1M for gpt-4o) |
| Cost per query | Sum of all model calls for one question (~$0.004) |

**Code:** `src/agent/agent.py` + `src/embedding/embedder.py`

---

## Evaluation

**What:** Running the pipeline against a golden set of questions and scoring the answers.
**Why:** You cannot improve what you do not measure. Evaluation catches regressions when you change chunking, retrieval, or generation parameters.
**Metrics:**
- **Faithfulness:** Did the answer stick to the retrieved context?
- **Context recall:** Did retrieval find the relevant chunks?
- **Context precision:** Were the retrieved chunks actually useful?
- **Answer relevancy:** Did the answer address the question?

**Tool:** RAGAS + DeepEval. GPT-4o acts as judge.
**Code:** `tests/test_eval.py`, `scripts/build_golden_set.py`

---

## CI Pipeline

**What:** Automated checks that run on every push and pull request.
**Why:** Catches bugs before they reach main. No LLM calls — fast and free.
**Steps:**
1. `ruff check` — linting and import sorting
2. `mypy` — static type checking
3. `pytest` — unit tests with coverage report

**Code:** `.github/workflows/ci.yml`
