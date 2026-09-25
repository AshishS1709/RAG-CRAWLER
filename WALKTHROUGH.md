# Walkthrough — Website-Grounded RAG Agent

This document provides a detailed walkthrough of the architecture, implementation choices, retrieval and grounding approach, example queries, evaluation methodology, cost analysis, and future improvements for the Website-Grounded RAG Agent.

---

## 1. Architecture Overview

The system is split into two distinct phases: **Ingestion** (offline, one-time per website) and **Per-Query RAG** (real-time, per user question).

### Ingestion Pipeline

The ingestion pipeline takes any public URL as input and produces a searchable vector knowledge base with zero site-specific logic:

1. **BFS Web Crawler** (`crawler/web_crawler.py`) — A breadth-first crawler built on `requests` and `BeautifulSoup`. It respects `robots.txt` via Python's `RobotFileParser`, stays scoped to the seed domain's path prefix (preventing spidering across unrelated sites), and enforces configurable `max_pages`, `max_depth`, and `crawl_delay` limits. Each crawled page is stored as a `CrawledPage` dataclass containing the URL, title, raw HTML, HTTP status, depth, and discovered links.

2. **Content Extraction & Chunking** (`crawler/content_processor.py`) — Raw HTML is processed through a two-tier extraction strategy: Trafilatura is the primary extractor (best-in-class for stripping navigation, footers, sidebars, and boilerplate), with a BeautifulSoup fallback that targets semantic HTML5 containers (`<main>`, `<article>`, `[role="main"]`). Extracted text is normalised and split using LangChain's `RecursiveCharacterTextSplitter` with 800-character chunks and 150-character overlap. Each chunk becomes a LangChain `Document` with metadata preserving the source URL, page title, chunk index, and total chunk count.

3. **Local Embedding** (`vectorstore/embeddings.py`) — Chunks are embedded using `BAAI/bge-small-en-v1.5` running entirely locally via SentenceTransformers and PyTorch. The model produces 384-dimensional dense vectors with top-tier retrieval quality on the MTEB benchmark. Device selection (CUDA, MPS, CPU) is automatic. No API key is required, and there is zero cost at any scale.

4. **Vector Storage** (`vectorstore/chroma_store.py`) — Embedded vectors are persisted in ChromaDB on local disk, organised into named collections. The wrapper supports add, search, stats, and reset operations.

### Per-Query RAG Workflow

Each user question is processed through a **LangGraph stateful graph** (`agent/graph.py`) with five nodes and conditional routing:

```
START → rewrite_query → retrieve → grade_documents → [conditional]
                                                       ├── generate → END
                                                       └── handle_no_docs → END
```

1. **rewrite_query** — The LLM (Groq) reformulates the user's question into a clearer, more specific search query optimised for retrieval. For example, "how do I make a filtered list in one line?" becomes "Python list comprehension with conditional filtering syntax". This step measurably improves recall for paraphrased or vague questions.

2. **retrieve** — The rewritten query is embedded and used for similarity search against ChromaDB, returning the top-k (default: 6) most relevant chunks.

3. **grade_documents** — Each retrieved chunk is individually evaluated by the LLM for relevance to the original question. The grader returns a JSON `{"score": "yes"}` or `{"score": "no"}` per chunk. Irrelevant chunks are filtered out before generation.

4. **Conditional routing** — If at least one chunk passes grading, the workflow routes to `generate`. If zero chunks pass, it routes to `handle_no_docs` for a graceful refusal.

5. **generate** — The LLM synthesises a grounded answer using only the relevant chunks as context, with strict instructions to never use outside knowledge. Source URLs are extracted and deduplicated for citation.

6. **handle_no_docs** — Returns a clear refusal message: "I don't have enough information in the documentation knowledge base to answer this question."

---

## 2. Key Implementation Choices

### Why LangGraph instead of a simple LLM chain?

The conditional routing after grading — generate if relevant docs exist, refuse if they don't — cannot be expressed cleanly in a linear `LLMChain`. LangGraph provides explicit, inspectable state transitions with a typed `RAGState` dictionary, making the workflow debuggable and extensible. Adding a new node (e.g., re-retrieval with a fallback query, or a web search fallback) requires adding a single node and edge, not restructuring the entire pipeline.

### Why local embeddings (BAAI/bge-small-en-v1.5) instead of OpenAI?

- **Cost**: $0.00 at any scale. No API key, no per-token billing. For comparison, embedding the same 80-page corpus through OpenAI's `text-embedding-3-small` would cost ~$0.003 — negligible but non-zero. The local model has zero marginal cost and works fully offline after a one-time ~133 MB download.
- **Quality**: BGE-small-en-v1.5 achieves top-tier scores on the MTEB retrieval benchmark, competitive with models 3-5x its size.
- **Privacy**: All data stays local. No content is sent to external APIs during ingestion.

### Why LLM-based document grading instead of cosine similarity thresholds?

A cosine similarity threshold is corpus-dependent and hard to tune. A threshold of 0.8 might pass a chunk about "Python 2 list syntax" when the user asks about "Python 3 generators" — both score high geometrically, but the LLM catches the semantic mismatch. LLM grading understands intent, not just surface similarity.

The grader defaults to `"yes"` on JSON parse errors (when the LLM wraps its response in markdown code fences or adds commentary). This is an intentional recall-over-precision tradeoff: it is better to pass a noisy chunk than to crash the pipeline.

### Why Trafilatura for content extraction?

Raw website HTML contains navigation bars, footers, sidebars, cookie banners, and JavaScript noise. Embedding this boilerplate produces vectors that pollute retrieval with irrelevant matches. Trafilatura uses heuristic and structural analysis to extract only the main article content, producing significantly cleaner text for embedding. A BeautifulSoup fallback targets semantic HTML5 containers (`<main>`, `<article>`, `[role="main"]`) when Trafilatura's extraction is sparse.

### Why query rewriting?

Users rarely phrase questions in the exact terminology used in documentation. "How do I make a filtered list in one line?" will not match "list comprehension" via embedding similarity as strongly as an explicit rewrite would. The rewrite step bridges this vocabulary gap by expanding the user's question with domain-relevant terms before it hits the vector store.

### Why Groq for LLM inference?

Groq provides ultra-fast inference on their custom LPU hardware. Using `openai/gpt-oss-20b` at $0.075/$0.30 per million input/output tokens, each query costs under $0.001. The free tier is sufficient for development and demo workloads. The model shows strong instruction-following for structured tasks like JSON grading.

---

## 3. Retrieval and Grounding Approach

### Strict Grounding via Prompt Design

The core anti-hallucination mechanism is the generation system prompt (`agent/prompts.py`):

> "Answer the user's question using ONLY the information provided in the context below. Base your answer EXCLUSIVELY on the provided context. Do not use any outside knowledge. If the context does not contain enough information to answer the question, say: 'I don't have enough information in the provided documentation to answer this question.' Do NOT speculate or fill gaps with assumed knowledge."

This creates a hard boundary: the LLM can only synthesise from the retrieved chunks. If the chunks don't cover the topic, the model must refuse rather than draw on its training data.

### Permissive Grading for Recall

The grading prompt is intentionally permissive:

> "Be permissive: if the document contains ANY information that could help answer the question, even partially, score it 'yes'."

This maximises recall at the cost of slightly noisier context. The rationale is that the generation step handles precision — it can ignore marginally relevant context — but it cannot generate accurate answers from chunks that were filtered out. Missing a relevant chunk is worse than including a noisy one.

### Graceful Refusal

When the grader marks all retrieved chunks as irrelevant, the system returns a clear refusal with a suggestion to rephrase. This is critical for trust: a RAG system that hallucinates when it has no relevant context is worse than one that honestly says "I don't know."

### Two-Tier Citation Tracking

The pipeline tracks provenance at two levels:
- **Context Sources**: All graded-relevant chunks supplied in the LLM context window, listed for full auditability.
- **In-Text Citations**: The specific subset of sources the model's generated answer directly drew upon.

---

## 4. Example Queries

The system has been verified against two structurally different sites — `docs.python.org/3/` (Sphinx) and `fastapi.tiangolo.com/tutorial/` (MkDocs Material) — without any code changes.

### Example 1: In-Scope Factual Query (FastAPI Collection)

**Question:** "How do path parameters work in FastAPI?"  
**Rewritten Query:** "FastAPI path parameters definition and usage in Python"  
**Answer:** "Path parameters in FastAPI are defined using Python format string syntax within the path decorator (e.g. `@app.get("/items/{item_id}")`). The parameter's type can be declared using standard Python type annotations (e.g., `item_id: int`), which FastAPI uses for automatic request parsing, data validation, and interactive documentation."  
**Sources:** `https://fastapi.tiangolo.com/tutorial/path-params/`, `https://fastapi.tiangolo.com/tutorial/`  
**Token/Cost:** 3,613 prompt + 248 completion = ~$0.000345

### Example 2: Paraphrased Query

**Question:** "How can I create a filtered list in a single line of Python code?"  
**Expected behaviour:** The rewrite step transforms this into something like "Python list comprehension with conditional filtering". The system retrieves the same tutorial/datastructures pages as the direct "what is a list comprehension" query, demonstrating retrieval robustness under paraphrasing.

### Example 3: Out-of-Scope Query (Correct Refusal)

**Question:** "What is Elon Musk's net worth?"  
**Expected behaviour:** The grader marks all 6 retrieved chunks as irrelevant (score: "no"). The system routes to `handle_no_docs` and returns: "I don't have enough information in the documentation knowledge base to answer this question."

### Example 4: Misleading Premise

**Question:** "Since Python lists are immutable, how do I modify them?"  
**Expected behaviour:** The system should retrieve relevant chunks about Python lists and correct the false premise — Python lists are mutable, not immutable — while providing grounded information about list modification methods.

### Example 5: Cross-Site Generality (FastAPI)

**Question:** "How do I configure gRPC streaming in FastAPI?"  
**Rewritten Query:** "configure gRPC streaming FastAPI documentation"  
**Document Grading:** 0 of 6 retrieved chunks relevant.  
**Answer:** "I could not find information about configuring gRPC streaming in the provided FastAPI tutorial documentation. The documentation focuses on REST APIs using HTTP methods (GET, POST, etc.) and does not cover gRPC streaming."

This demonstrates correct refusal on a topic that is related to the indexed site but not covered in the crawled content.

---

## 5. Evaluation Methodology

### Question Design

The evaluation suite (`evaluation/eval_questions.json`) contains 15 questions across 5 categories designed to test different retrieval and reasoning challenges:

| Category | Count | Tests |
|---|---|---|
| **Straightforward** | 4 | Basic factual retrieval (list comprehension, `len()`, file I/O, decorators) |
| **Paraphrased** | 2 | Same concepts as straightforward, different wording — tests retrieval robustness |
| **Multi-page** | 4 | Require combining information across multiple documentation pages (`__str__` vs `__repr__`, generators, context managers, Python 3.10 features) |
| **Misleading** | 2 | False premises the agent should correct (pass-by-value misconception, list immutability) |
| **Unanswerable** | 3 | Completely out of scope — should refuse (JavaScript frameworks, celebrity net worth, Python 4.0) |

### Scoring

The automated evaluation harness (`evaluation/run_eval.py`) runs the full RAG pipeline on each question and uses an LLM-as-judge (same Groq model) to score each answer on two dimensions:
- **Faithfulness** (0–5): Is the answer consistent with the retrieved context? Does it hallucinate?
- **Relevance** (0–5): Does the answer address the user's question?

For unanswerable questions, the harness checks refusal accuracy — did the system correctly decline rather than fabricate an answer?

For answerable questions, keyword recall is checked against expected answer keywords.

---

## 6. Cost Analysis

### Ingestion Cost (One-Time)

| Item | Value | Cost |
|---|---|---|
| Pages crawled | 80 (configurable) | — |
| Chunks produced | ~640 | — |
| Embedding model | BAAI/bge-small-en-v1.5 (local) | **$0.00** |
| **Total ingestion** | — | **$0.00** |

Ingestion is entirely free because the embedding model runs locally. There are no API calls, no API key requirements, and no per-token billing.

### Per-Query Cost (Groq `openai/gpt-oss-20b`)

Each query makes three separate Groq LLM calls:

| Step | Prompt Tokens | Completion Tokens | Cost (est.) |
|---|---|---|---|
| Query rewrite | ~150 | ~50 | ~$0.000026 |
| Document grading (6 chunks) | ~2,400 | ~60 | ~$0.000198 |
| Answer generation | ~1,800 | ~300 | ~$0.000225 |
| **Total per query** | **~4,350** | **~410** | **~$0.000449** |

### Scale Projections

| Scale | Ingestion | Query Cost | **Total** |
|---|---|---|---|
| Demo (15 eval queries) | $0.00 | ~$0.007 | **~$0.007** |
| 100 queries | $0.00 | ~$0.045 | **~$0.045** |
| 1,000 queries | $0.00 | ~$0.449 | **~$0.449** |
| 10,000 queries | $0.00 | ~$4.49 | **~$4.49** |

### Cost Optimisation Levers

| Optimisation | Expected Savings | Trade-off |
|---|---|---|
| Reduce `top_k` from 6 to 4 | ~33% on grading step | Slightly lower recall |
| Use `openai/gpt-oss-20b` vs `120b` | ~2-4× cheaper | Slightly lower quality |
| Replace LLM grading with cosine threshold | Eliminates grading cost (~40-50%) | Higher hallucination risk |
| Cache repeated queries | Up to 100% on hot queries | Requires Redis or equivalent |

---

## 7. Security Considerations

1. **Indirect Prompt Injection**: Crawled website content is treated as untrusted data but is passed into the LLM context window without active sanitisation. The strict grounding prompt constrains the model to use retrieved chunks as factual source material rather than executing instructions found within context. This is a partial mitigation, not a complete defense. A production system would require a dedicated prompt-injection classifier or dual-LLM architecture.

2. **Input Validation**: The dynamic ingestion UI enforces server-side validation — only `http://` and `https://` schemes with valid hostnames are accepted. Dangerous schemes (`file:`, `javascript:`, `data:`) are strictly rejected. Max pages is hard-clamped server-side at 50 regardless of UI input.

---

## 8. Known Limitations

1. **JSON parsing in grading**: LLMs occasionally wrap JSON in markdown code fences. The grader defaults to "yes" on parse errors, which may pass slightly noisy chunks.
2. **Single-domain crawl scope**: Intentionally scoped to prevent unbounded spidering. Very large sites may need higher `max_pages` and `max_depth`.
3. **First-run model download**: Requires ~133 MB download from HuggingFace. Subsequent runs are fully offline.
4. **No JavaScript rendering**: `requests` + Trafilatura extract server-rendered HTML only. SPAs with client-side rendering will yield empty extractions.
5. **Local ChromaDB concurrency**: Not safe for multi-process concurrent writes.
6. **No rate limiting or auth**: No token-bucket rate limits or user authentication on the UI endpoints.

---

## 9. What I Would Improve Next

1. **Hybrid retrieval**: Combine BM25 sparse search with dense vector search using reciprocal rank fusion. This improves precision for exact function names (e.g., `itertools.groupby`) where keyword match outperforms embedding similarity.

2. **Hierarchical chunking**: Implement a parent-child chunk strategy — retrieve small chunks for precision, expand to the parent chunk for full context during generation.

3. **Streaming responses**: Stream Groq output token-by-token to the Streamlit UI for better perceived responsiveness.

4. **Cross-encoder re-ranking**: Add a re-ranker pass between retrieval and generation for higher precision, especially with borderline-relevant chunks.

5. **Async crawling**: Replace the synchronous `requests`-based crawler with `aiohttp` + `asyncio` for 5–10× faster ingestion.

6. **Persistent evaluation tracking**: Store eval results across runs with timestamps to detect quality regressions as prompts or models change.
