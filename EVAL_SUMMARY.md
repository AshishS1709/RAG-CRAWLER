# Evaluation Summary — Website-Grounded RAG Agent

## Evaluation Design

The evaluation suite (`evaluation/eval_questions.json`) contains **15 questions** across 5 categories, each targeting a different retrieval and reasoning challenge:

| Category | Questions | What it Tests |
|---|---|---|
| **Straightforward** | Q01, Q02, Q03, Q14 | Basic factual retrieval — directly matches documentation content |
| **Paraphrased** | Q04, Q05 | Same concepts reworded — tests retrieval robustness under vocabulary mismatch |
| **Multi-page** | Q06, Q07, Q08, Q15 | Requires combining information from multiple documentation pages |
| **Misleading** | Q09, Q10 | False premises — agent should correct the misconception, not agree with it |
| **Unanswerable** | Q11, Q12, Q13 | Completely out of scope — agent should refuse rather than hallucinate |

## Scoring Methodology

The automated harness (`evaluation/run_eval.py`) scores each answer on:

- **Faithfulness** (0–5): Is the answer consistent with retrieved context? Does it hallucinate?
- **Relevance** (0–5): Does the answer address the user's actual question?
- **Refusal accuracy** (for unanswerable): Did the system correctly decline?
- **Keyword recall** (for answerable): Does the answer contain expected keywords?

Scoring uses an **LLM-as-judge** approach (same Groq model) for faithfulness and relevance.

## Cross-Site Verification Results (FastAPI)

The pipeline was independently verified against FastAPI's tutorial documentation (MkDocs Material, structurally different from Python docs' Sphinx layout) — **zero code changes required**:

| Metric | Result |
|---|---|
| Pages crawled | 20 |
| Chunks indexed | 254 |
| Ingestion cost | $0.00 |
| Collection | `fastapi_demo` |

### In-Scope Query — Grounded Answer

**Q:** "How do path parameters work in FastAPI?"
**Rewritten Query:** "FastAPI path parameters definition and usage in Python"
**Answer:** Correctly described `@app.get("/items/{item_id}")` syntax with type annotations for automatic parsing, validation, and documentation.
**Sources:** `fastapi.tiangolo.com/tutorial/path-params/`, `fastapi.tiangolo.com/tutorial/`
**Cost:** 3,613 prompt + 248 completion = ~$0.000345

### Out-of-Scope Query — Correct Refusal

**Q:** "How do I configure gRPC streaming in FastAPI?"
**Grading:** 0 of 6 retrieved chunks deemed relevant.
**Answer:** Correctly refused — stated the documentation covers REST APIs only, not gRPC streaming.

## Expected Behaviour by Category

| Category | Expected Outcome | Key Metric |
|---|---|---|
| **Straightforward** | High faithfulness and relevance scores (4-5) | Direct keyword matches in answers |
| **Paraphrased** | Same quality as straightforward — rewrite step bridges vocabulary gap | Retrieves same source pages as the direct question |
| **Multi-page** | Moderate-to-high scores — synthesis across multiple chunks | Multiple source URLs cited |
| **Misleading** | Agent corrects the false premise with grounded evidence | Does not agree with the incorrect assertion |
| **Unanswerable** | Clean refusal — no hallucinated answer | Refusal accuracy = 100% |

## Key Design Insight

The **query rewriting step** is critical for paraphrased questions. Without it, "How can I create a filtered list in a single line of Python code?" would retrieve weaker matches than "What is a list comprehension?" — because the raw text doesn't contain the term "comprehension." The rewrite step bridges this gap before retrieval.

The **LLM-based document grader** is critical for unanswerable questions. Even when ChromaDB returns 6 chunks (it always returns `top_k` results regardless of relevance), the grader correctly marks all of them as irrelevant for questions like "What is Elon Musk's net worth?" — routing the query to the refusal path instead of forcing the generator to hallucinate from unrelated Python documentation.

## Running the Evaluation

```bash
# Run all 15 questions:
python evaluation/run_eval.py

# Save full JSON results:
python evaluation/run_eval.py --output evaluation/eval_results.json
```

Requires a populated ChromaDB collection (run `python ingest.py` first) and a valid `GROQ_API_KEY`.
