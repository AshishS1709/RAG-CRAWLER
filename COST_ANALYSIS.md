# Cost Analysis — Website-Grounded RAG Agent

---

## 1. Ingestion (One-Time)

### What happens during ingestion?

1. **Crawl** — `requests` fetches each page (no LLM involved, no API cost)
2. **Extract & Chunk** — `trafilatura` + `RecursiveCharacterTextSplitter` (no LLM involved)
3. **Embed** — `BAAI/bge-small-en-v1.5` runs **locally on your machine**

| Metric | Value |
|---|---|
| Pages crawled | 80 (default configuration) |
| Chunks produced | ~640 (~8 chunks per page average) |
| Embedding tokens | ~128,000 (~200 tokens per chunk × 640 chunks) |
| Embedding API calls | 0 (local inference) |
| **Ingestion cost** | **$0.00** |

### Why is ingestion free?

The embedding model — `BAAI/bge-small-en-v1.5` — runs entirely via **SentenceTransformers**
on local hardware (CPU, CUDA GPU, or Apple MPS). There is no API call,
no API key, and no per-token billing.

For comparison, embedding the same corpus through the OpenAI Embeddings API
(`text-embedding-3-small` at $0.02/1M tokens) would cost roughly $0.003 for ~128K tokens
— negligible but non-zero. The local model has zero marginal cost at any scale and works
fully offline after the one-time model download.

---

## 2. Per-Query Cost — Worked Example

Each query invokes **three LLM calls** on Groq (`openai/gpt-oss-20b`):

**Groq pricing for `openai/gpt-oss-20b`:**
- Input: $0.075 / 1M tokens
- Output: $0.30 / 1M tokens
- Source: https://groq.com/pricing/

### Worked example: "How do path parameters work in FastAPI?"

| Step | Prompt Tokens | Completion Tokens | Input Cost | Output Cost | Step Total |
|---|---|---|---|---|---|
| **Query rewrite** | ~150 | ~50 | $0.000011 | $0.000015 | **$0.000026** |
| **Document grading** (6 chunks × ~400 tokens each) | ~2,400 | ~60 | $0.000180 | $0.000018 | **$0.000198** |
| **Answer generation** | ~1,800 | ~300 | $0.000135 | $0.000090 | **$0.000225** |
| **Total per query** | **~4,350** | **~410** | **$0.000326** | **$0.000123** | **$0.000449** |

> **Observed from FastAPI demo query:** 3,613 prompt tokens + 248 completion tokens = ~$0.000345
> for a grounded answer with 2 source citations.

### Cost per step — where the money goes

- **Grading dominates**: Document grading is the most expensive step (~44% of total cost) because it makes one LLM call per retrieved chunk (6 calls for top-k=6).
- **Generation is second**: ~50% of cost, but this is the step that produces the user-facing answer.
- **Rewriting is cheap**: ~6% of cost for a high-impact retrieval improvement.

---

## 3. Scale Projections

Based on an average cost of **~$0.000449 per query** (Groq `openai/gpt-oss-20b`):

| Scale | Ingestion | Query Cost | **Total** |
|---|---|---|---|
| Demo (15 eval queries) | $0.00 | ~$0.007 | **~$0.007** |
| 100 queries | $0.00 | ~$0.045 | **~$0.045** |
| 1,000 queries | $0.00 | ~$0.449 | **~$0.449** |
| 10,000 queries | $0.00 | ~$4.49 | **~$4.49** |

> At the free Groq tier, the entire demo workload (ingestion + 15 eval queries + interactive testing)
> costs well under $0.01 in LLM inference. Ingestion is always $0.00.

---

## 4. Cost Breakdown Notes (Architecture)

### Why three LLM calls per query?

The RAG pipeline makes separate Groq API calls at each reasoning step:

1. **Query rewrite** — Small call, big impact: an expanded query significantly
   improves retrieval recall for paraphrased or vague questions.

2. **Document grading** — This is typically the most expensive step per query
   because it runs one LLM call per retrieved chunk (`top_k` calls total). It is
   also the most impactful for reducing hallucination: an LLM catches semantic
   mismatches that cosine similarity cannot.

3. **Generation** — Final synthesis step; cost scales with context size (number
   of relevant chunks passed in) and answer length.

### Optimisation levers

| Optimisation | Expected Savings | Trade-off |
|---|---|---|
| Reduce `top_k` from 6 to 4 | ~33% on grading | Slightly lower recall |
| Use `openai/gpt-oss-20b` vs `120b` | ~2-4× cheaper | Slightly lower quality |
| Replace LLM grading with similarity threshold | Eliminates grading cost (~40-50%) | Higher hallucination risk |
| Cache repeated queries | Up to 100% on hot queries | Requires Redis or equivalent |
