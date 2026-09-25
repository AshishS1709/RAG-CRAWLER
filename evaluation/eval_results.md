# Evaluation Results

## Overview

| Metric | Value |
|---|---|
| Total questions | 15 |
| Categories | 5 (straightforward, paraphrased, multi-page, misleading, unanswerable) |
| Answerable questions | 12 (should produce grounded answers) |
| Unanswerable questions | 3 (should refuse) |
| Scoring method | LLM-as-judge (faithfulness 0–5, relevance 0–5) + keyword recall |

## Question Breakdown

| ID | Category | Question | Should Answer | Expected Key Concepts |
|---|---|---|---|---|
| Q01 | Straightforward | What is a list comprehension in Python? | Yes | list comprehension, iterable, condition |
| Q02 | Straightforward | What does the built-in len() function do? | Yes | length, object, sequence |
| Q03 | Straightforward | How do you open a file in Python? | Yes | open, mode, with |
| Q04 | Paraphrased | How can I create a filtered list in a single line? | Yes | comprehension, for, if |
| Q05 | Paraphrased | What function tells me how many elements are in a Python object? | Yes | len |
| Q06 | Multi-page | Difference between \_\_str\_\_ and \_\_repr\_\_? | Yes | \_\_str\_\_, \_\_repr\_\_, representation |
| Q07 | Multi-page | How do Python generators differ from regular functions? | Yes | yield, iterator, lazy, memory |
| Q08 | Multi-page | How does Python's context manager protocol work? | Yes | \_\_enter\_\_, \_\_exit\_\_, with |
| Q09 | Misleading | Python uses pass-by-value for all objects, right? | Yes | pass-by-object, reference, mutable |
| Q10 | Misleading | Since Python lists are immutable, how do I modify them? | Yes | mutable, append, not immutable |
| Q11 | Unanswerable | What is the best JavaScript framework for building React apps? | No | Should refuse |
| Q12 | Unanswerable | What is Elon Musk's net worth? | No | Should refuse |
| Q13 | Unanswerable | What is the Python release date for version 4.0? | No | Should refuse |
| Q14 | Straightforward | What are Python decorators and how do they work? | Yes | decorator, function, wrap, @ |
| Q15 | Multi-page | What new features were introduced in Python 3.10? | Yes | match, structural pattern matching |

## Cross-Site Verification (FastAPI Demo)

The same pipeline was verified against FastAPI documentation (20 pages, 254 chunks) with no code changes:

- **In-scope query** ("How do path parameters work?"): Produced a grounded answer citing `fastapi.tiangolo.com/tutorial/path-params/` with correct technical details about type annotations and automatic validation.
- **Out-of-scope query** ("How do I configure gRPC streaming?"): Correctly refused — 0/6 chunks passed grading, returned a clear "not found in documentation" response.

## How to Run

```bash
# Prerequisite: ingest a website first
python ingest.py --url https://docs.python.org/3/ --max-pages 80 --collection python_docs

# Run evaluation
python evaluation/run_eval.py

# Save detailed JSON results
python evaluation/run_eval.py --output evaluation/eval_results.json
```

See [`EVAL_SUMMARY.md`](../EVAL_SUMMARY.md) for methodology and analysis.
