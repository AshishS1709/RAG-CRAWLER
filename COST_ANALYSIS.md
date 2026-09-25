# Cost Analysis — Python Docs RAG Agent

> ⚠️ **Sections 2 and 3 (Worked Example and Scale Projections) are PENDING a real run.**
> Once the real eval run completes and `eval_results.json` is populated, the actual
> average cost-per-query will be used to fill in real numbers here.

---

## 1. Ingestion (One-Time)

### What happens during ingestion?

1. **Crawl** — `requests` fetches each page (no LLM involved, no API cost)
2. **Extract & Chunk** — `trafilatura` + `RecursiveCharacterTextSplitter` (no LLM involved)
3. **Embed** — `nomic-ai/nomic-embed-text-v1` runs **locally on your machine**

| Metric | Source |
|---|---|
| Pages crawled | See `evaluation/ingestion_run_log.txt` (real run) |
| Chunks produced | See `evaluation/ingestion_run_log.txt` (real run) |
| Embedding tokens | See `evaluation/ingestion_run_log.txt` — `usage_report()` output |
| Embedding API calls | 0 |
| **Ingestion cost** | **$0.00** |

### Why is ingestion free?

The embedding model — `nomic-ai/nomic-embed-text-v1` — runs entirely via **HuggingFace
Transformers** on local hardware (CPU, CUDA GPU, or Apple MPS). There is no API call,
no API key, and no per-token billing.

For comparison, embedding the same corpus through the OpenAI Embeddings API
(`text-embedding-3-small` at $0.02/1M tokens) would cost roughly $0.003 for ~150K tokens
— negligible but non-zero. The local model has zero marginal cost at any scale and works
fully offline after the one-time model download.

---

## 2. Per-Query Cost — Worked Example

**PENDING REAL RUN.**

The per-query cost calculation requires real token counts from an actual query.

Each query invokes **three LLM calls** on Groq:

| Step | What it does | Notes |
|---|---|---|
| Query rewrite | Expands/clarifies the user question | ~150 input tokens, ~50 output tokens |
| Document grading | `top_k` (default: 6) calls, one per retrieved chunk | Most expensive step per query |
| Answer generation | Synthesises answer from relevant chunks | Scales with chunk count and answer length |

**Groq pricing for `llama-3.1-8b-instant`:**
- Input: $0.05 / 1M tokens
- Output: $0.08 / 1M tokens
- Source: https://groq.com/pricing/

The real per-query cost with actual token counts will be inserted here after the eval run.

---

## 3. Scale Projections

**PENDING REAL RUN.**

The projection table will be based on the actual average cost-per-query observed in the
real eval run:

```
avg_cost_per_query = eval_results.json["summary"]["total_cost_usd"]
                     / eval_results.json["summary"]["total_questions"]
```

| Scale | Ingestion | Per-Query Cost | Total |
|---|---|---|---|
| 15 (eval suite) | $0.00 | (real number × 15) | TBD |
| 100 queries | $0.00 | (real number × 100) | TBD |
| 1,000 queries | $0.00 | (real number × 1,000) | TBD |
| 10,000 queries | $0.00 | (real number × 10,000) | TBD |

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
| Use `llama-3.1-8b-instant` vs `70b` | ~12× cheaper | Slightly lower quality |
| Replace LLM grading with similarity threshold | Eliminates grading cost (~40-50%) | Higher hallucination risk |
| Cache repeated queries | Up to 100% on hot queries | Requires Redis or equivalent |
