# helix-rag

An agentic RAG pipeline you can point at any collection of research papers. Drop in your PDFs, ask questions, get grounded answers with source citations.

Built as a portfolio project to demonstrate a production-grade RAG system end-to-end: ingestion, hybrid search, reranking, evaluation, and an agent wrapper. The example use case is biomarker research papers, but the pipeline works for any domain.

---

## What this actually does

Most RAG tutorials show you how to build a basic similarity search. This project goes further:

- Hybrid search (keyword + semantic) so exact terms like gene names and paper titles are not missed
- A reranker that re-scores results before sending to the LLM, not just raw vector similarity
- An agent wrapper so the RAG pipeline becomes a tool the agent can call (or not call, if it does not need to)
- Evaluation using RAGAS and DeepEval with a real quality gate in CI
- Guardrails so the system says "I don't know" instead of making things up

The architecture is deliberately modular. Swapping the vector DB, the embedding model, or the LLM requires changing one file, not rewriting the system.

---

## How it works

There are two phases.

**Building the index** (run once, or when you add new papers):

```
PDFs -> extract text -> split into chunks -> embed -> store in Qdrant
```

**Answering questions** (runs on every user query):

```
Question -> embed -> hybrid search -> rerank top results -> GPT-4o -> answer + sources
```

The reason for splitting it this way: you only pay to embed your documents once. After that, querying is cheap and fast.

---

## Project structure

```
helix-rag/
  data/
    raw/              <- Drop your PDFs here (not tracked by git)
    processed/        <- Parsed text output (not tracked by git)
  src/
    ingestion/        <- Extracts text from PDFs
    chunking/         <- Splits text into searchable pieces
    embedding/        <- Wrapper around the embedding model
    vectorstore/      <- Wrapper around Qdrant
    retrieval/        <- Hybrid search + reranker
    generation/       <- Prompt builder + GPT-4o call
    agent/            <- OpenAI Agents SDK: RAG as a callable tool
    ui/               <- Gradio chat interface
  evals/              <- RAGAS and DeepEval scripts
  scripts/            <- CLI for ingestion and querying
  tests/              <- Unit tests (no live API calls)
  .github/workflows/  <- CI: runs lint, tests, and eval gate automatically
```

---

## Architecture decisions

Every choice below has an alternative. If your company uses a different stack, the interface is the same, just swap the implementation.

### PDF parsing: PyMuPDF

Scientific papers are often two-column PDFs. Most parsers merge the two columns into garbage text. PyMuPDF handles layout correctly and is fast enough to process hundreds of papers in seconds.

The alternative is `pdfplumber`, which is better at extracting tables but slower on large documents. If your papers have a lot of data tables you care about, swap it in.

### Chunking: recursive splitting with parent-child storage

This is the part most tutorials skip, and it matters a lot for answer quality.

Small chunks (around 300 tokens) score better in similarity search because they are focused on one idea. But 300 tokens is often too little context for the LLM to write a good answer. Parent-child chunking solves this:

```
Full section (1200 tokens) = parent, stored separately

  Child chunk A (300 tokens)  <- what gets searched
  Child chunk B (300 tokens)  <- what gets searched
  Child chunk C (300 tokens)  <- what gets searched
  Child chunk D (300 tokens)  <- what gets searched

User asks a question -> matches Child B -> LLM receives the full parent (1200 tokens)
```

The search is precise. The context sent to the LLM is rich.

The alternative is semantic chunking, which splits on topic shifts rather than character count. It is smarter but significantly slower and more expensive to run.

### Embedding: text-embedding-3-small

OpenAI's embedding models are easy to start with and strong enough for most use cases. `text-embedding-3-small` costs about $0.02 per million tokens. For 10 to 20 research papers, the total ingestion cost is under a cent.

The main alternative is `BAAI/bge-large-en-v1.5`, a free open-source model you run locally. It is slightly stronger on technical text but requires a GPU to run at reasonable speed. The codebase supports swapping it in by changing one environment variable.

### Vector database: Qdrant

Qdrant is used in production at many companies and runs locally in Docker with no account required. It supports hybrid search natively (meaning it handles both keyword and vector search in a single call, rather than you managing two separate systems).

The main alternative you will hear about in interviews is Pinecone, which is fully managed and has high name recognition. The tradeoff is that it is cloud-only and requires a paid account at scale. The vector store module is abstracted so swapping to Pinecone requires changing only `src/vectorstore/`.

### Retrieval: hybrid search followed by reranking

Dense vector search is good at finding semantically similar content. But it sometimes misses exact terms, which matters a lot for technical domains where specific names and identifiers are important.

Hybrid search combines dense vector search with BM25 keyword search. The scores are merged and you get the best of both.

After hybrid search returns the top 20 candidates, a cross-encoder reranker re-scores each one. Unlike the embedding model which scores query and chunk separately, the cross-encoder reads them together, which gives more accurate relevance scores. The top 5 go to the LLM.

```
Question
  -> hybrid search  ->  top 20 candidates
  -> cross-encoder reranker  ->  top 5
  -> GPT-4o  ->  answer
```

### Generation: GPT-4o with a grounding constraint

The prompt explicitly tells GPT-4o to answer only from the retrieved chunks. If the answer is not there, it says so. This is the primary mechanism for preventing hallucination.

```
Answer only using the provided context.
If the context does not contain enough information, say "I don't have enough information to answer that."

Context: {retrieved chunks}
Question: {user question}
```

### Guardrails

Two layers are applied on every request.

Before retrieval: an input filter checks whether the question is on-topic for the indexed documents, and a prompt injection check blocks attempts to override the system instructions.

After generation: a citation check verifies that the answer references retrieved content. If no relevant chunks were found, the system returns a "not enough information" response rather than letting the LLM guess.

The library used is `guardrails-ai`. The enterprise alternative is NVIDIA NeMo Guardrails, which is more powerful but significantly more complex to configure.

### Evaluation: RAGAS and DeepEval

Three metrics are tracked:

- **Faithfulness**: does the answer stick to what the retrieved chunks actually say?
- **Context recall**: did the retrieval step find the chunks that actually contain the answer?
- **Context precision**: are the retrieved chunks relevant, or is there noise mixed in?

RAGAS is the most cited RAG benchmark framework. DeepEval runs the same metrics but integrates with pytest, which means the eval can run as part of CI and block a merge if quality drops.

A golden dataset of 30 to 40 (question, expected answer) pairs is used. The dataset is generated semi-automatically using RAGAS TestsetGenerator and then reviewed by hand.

The quality gate: if faithfulness drops below 0.80 on a merge to main, the CI job fails.

### Agent: OpenAI Agents SDK

The RAG pipeline is wrapped as a tool the agent can call. This makes it easy to add more tools later without changing the core pipeline.

```python
@tool
def search_papers(query: str) -> str:
    """Search the indexed research papers and return a grounded answer."""
    ...
```

The agent decides when to call the tool based on the user's question. It can also answer from its own knowledge when the question does not require the papers.

### UI: Gradio

Gradio is the standard way to demo ML systems. It takes about 20 lines of Python to build a functional chat interface. The alternative is a React frontend with a FastAPI backend, which is more impressive for a web engineering portfolio but significantly more code for the same result.

---

## Setup

You need Python 3.11+, Docker, and an OpenAI API key.

### 1. Clone the repo

```bash
git clone https://github.com/shalabhsuman/helix-rag.git
cd helix-rag
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

### 3. Configure your environment

```bash
cp .env.example .env
```

Open `.env` and set your `OPENAI_API_KEY`. Everything else can stay as the default to start.

### 4. Start Qdrant

Qdrant runs locally in Docker. This one command starts it and makes your data persist between restarts.

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Leave this running in a separate terminal tab while you use the pipeline.

---

## Ingesting papers

Drop your PDFs into `data/raw/`, then run:

```bash
python scripts/ingest.py --mode add
```

This parses every PDF, splits the text, embeds each chunk, and stores it in Qdrant. You will see progress output for each paper.

### Adding a new paper later

```bash
python scripts/ingest.py --mode add --input data/raw/new_paper.pdf
```

Only the new file is processed. Existing vectors are not touched.

### Removing a paper

```bash
python scripts/ingest.py --mode delete --doc_id author_year_title
```

### Rebuilding the entire index

Use this when you change the chunking strategy or switch embedding models. The old vectors are incompatible with the new ones, so a full rebuild is required.

```bash
python scripts/ingest.py --mode reindex
```

---

## Asking questions

```bash
python scripts/query.py --question "What is the relationship between this biomarker and tumor growth?"
```

Or run the chat UI:

```bash
python src/ui/app.py
```

Opens at `http://localhost:7860`.

---

## Evaluation

### Generating a golden dataset

The first time you set up evaluation, generate a synthetic question set from your papers:

```bash
python evals/generate_golden_dataset.py
```

This uses the RAGAS TestsetGenerator to create 50 question/answer pairs automatically. Review the output in `evals/golden_dataset.json`, remove any bad questions, and you have your benchmark dataset.

### Running the evaluation

```bash
python evals/run_ragas.py     # faithfulness, context recall, context precision
python evals/run_deepeval.py  # same metrics, outputs in pytest format
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

Unit tests mock all external calls (OpenAI, Qdrant). They run in under 2 minutes and cost nothing.

---

## CI

GitHub Actions runs automatically. No setup needed beyond adding your API keys as repository secrets.

| Workflow | When it runs | What it does |
|---|---|---|
| `ci.yml` | Every push | Lint, type check, unit tests |
| `eval.yml` | Merges to main | Full eval against golden dataset. Blocks merge if faithfulness < 0.80 |

To add your secrets: go to your GitHub repo > Settings > Secrets and variables > Actions > New repository secret. Add `OPENAI_API_KEY`.

---

## License

MIT
