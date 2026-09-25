"""
evaluation/run_eval.py

Automated evaluation harness for the RAG Agent.

Metrics:
  - Faithfulness   : Is the answer grounded in retrieved context? (LLM judge)
  - Answer relevance: Does the answer address the question?
  - Refusal accuracy: Did the agent correctly refuse unanswerable questions?
  - Source relevance: Are cited sources relevant?
  - Keyword recall  : Are expected answer keywords present?

Usage:
  python evaluation/run_eval.py
  python evaluation/run_eval.py --output results.json
  python evaluation/run_eval.py --questions-file evaluation/eval_questions.json
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Add parent dir to path so imports work when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()
logging.basicConfig(level=logging.WARNING)


# ── LLM Judge prompt ────────────────────────────────────────────────────────

FAITHFULNESS_JUDGE_PROMPT = """You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.

Evaluate the following answer for FAITHFULNESS — whether it is supported by the provided context,
and RELEVANCE — whether it correctly and completely addresses the question.

Question: {question}

Context provided to the agent:
---
{context}
---

Agent's answer:
---
{answer}
---

Respond with ONLY a JSON object with this exact structure:
{{
  "faithfulness": <0-5 integer>,
  "relevance": <0-5 integer>,
  "reasoning": "<one sentence>"
}}

Scoring guide (0–5):
  5: Perfect — answer fully supported by context, directly addresses question
  4: Good — minor gaps but mostly supported
  3: Adequate — partially supported, some unsupported claims
  2: Poor — significant unsupported claims or off-topic
  1: Bad — mostly unsupported or wrong
  0: Fail — completely unsupported or hallucinated
"""


# ── Result data classes ──────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    answer: str
    sources: List[str]
    rewritten_query: str
    should_answer: bool
    correctly_refused: Optional[bool]   # None if should_answer=True
    keyword_recall: float               # 0.0–1.0
    faithfulness_score: int             # 0–5
    relevance_score: int                # 0–5
    judge_reasoning: str
    prompt_tokens: int
    completion_tokens: int
    embedding_tokens: int
    total_cost_usd: float
    latency_s: float
    error: Optional[str] = None

    def passed(self) -> bool:
        """Simple pass/fail heuristic."""
        if not self.should_answer:
            return self.correctly_refused is True
        return self.faithfulness_score >= 3 and self.relevance_score >= 3


@dataclass
class EvalSummary:
    total_questions: int
    passed: int
    failed: int
    pass_rate: float
    avg_faithfulness: float
    avg_relevance: float
    avg_keyword_recall: float
    refusal_accuracy: float
    total_tokens: int
    total_cost_usd: float
    avg_latency_s: float
    results: List[QuestionResult] = field(default_factory=list)


# ── Evaluator ────────────────────────────────────────────────────────────────

class RAGEvaluator:
    def __init__(
        self,
        agent,
        judge_llm,
        questions_path: str = "evaluation/eval_questions.json",
    ):
        self.agent = agent
        self.judge = judge_llm
        self.questions_path = questions_path

    def _load_questions(self) -> List[Dict]:
        with open(self.questions_path) as f:
            data = json.load(f)
        return data["questions"]

    def _check_keyword_recall(self, answer: str, keywords: List[str]) -> float:
        if not keywords:
            return 1.0
        answer_lower = answer.lower()
        hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
        return hits / len(keywords)

    def _check_correctly_refused(self, answer: str) -> bool:
        """Did the agent say it doesn't have info?"""
        refuse_signals = [
            "don't have",
            "do not have",
            "not enough information",
            "cannot answer",
            "outside the scope",
            "not covered",
            "no information",
        ]
        answer_lower = answer.lower()
        return any(sig in answer_lower for sig in refuse_signals)

    def _llm_judge(self, question: str, answer: str, context: str) -> tuple[int, int, str]:
        """Run LLM-as-judge. Returns (faithfulness, relevance, reasoning)."""
        from langchain_core.messages import HumanMessage
        try:
            prompt = FAITHFULNESS_JUDGE_PROMPT.format(
                question=question, context=context[:3000], answer=answer
            )
            resp = self.judge.invoke([HumanMessage(content=prompt)])
            parsed = json.loads(resp.content.strip())
            return (
                int(parsed.get("faithfulness", 3)),
                int(parsed.get("relevance", 3)),
                parsed.get("reasoning", ""),
            )
        except Exception as e:
            return (3, 3, f"Judge error: {e}")

    def run(self) -> EvalSummary:
        questions = self._load_questions()
        results: List[QuestionResult] = []

        print(f"\n{'='*60}")
        print(f"  RUNNING EVALUATION ({len(questions)} questions)")
        print(f"{'='*60}\n")

        for q in questions:
            qid = q["id"]
            question = q["question"]
            category = q["category"]
            should_answer = q["should_answer"]
            expected_kws = q.get("expected_answer_contains", [])

            print(f"[{qid}] [{category.upper()}] {question[:70]}...")

            t0 = time.time()
            try:
                result = self.agent.ask(question)
                latency = time.time() - t0

                answer = result["answer"]
                sources = result.get("sources", [])
                rewritten = result.get("rewritten_query", "")
                usage = result["usage"]

                # Keyword recall
                kw_recall = self._check_keyword_recall(answer, expected_kws)

                # Refusal check (for unanswerable)
                correctly_refused = None
                if not should_answer:
                    correctly_refused = self._check_correctly_refused(answer)

                # LLM judge (skip for unanswerable correctly refused)
                if should_answer or not correctly_refused:
                    # Build a minimal context string from retrieved docs info
                    context_hint = f"Sources: {', '.join(sources)}" if sources else "No sources"
                    faith, relev, reasoning = self._llm_judge(question, answer, context_hint)
                else:
                    faith, relev, reasoning = 5, 5, "Correctly refused unanswerable question"

                qr = QuestionResult(
                    id=qid,
                    category=category,
                    question=question,
                    answer=answer,
                    sources=sources,
                    rewritten_query=rewritten,
                    should_answer=should_answer,
                    correctly_refused=correctly_refused,
                    keyword_recall=kw_recall,
                    faithfulness_score=faith,
                    relevance_score=relev,
                    judge_reasoning=reasoning,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    embedding_tokens=usage.embedding_tokens,
                    total_cost_usd=usage.total_cost_usd,
                    latency_s=latency,
                )

            except Exception as e:
                latency = time.time() - t0
                qr = QuestionResult(
                    id=qid, category=category, question=question,
                    answer="ERROR", sources=[], rewritten_query="",
                    should_answer=should_answer, correctly_refused=None,
                    keyword_recall=0.0, faithfulness_score=0, relevance_score=0,
                    judge_reasoning="", prompt_tokens=0, completion_tokens=0,
                    embedding_tokens=0, total_cost_usd=0.0, latency_s=latency,
                    error=str(e),
                )

            status = "✅" if qr.passed() else "❌"
            print(
                f"  {status} Faith={qr.faithfulness_score}/5  Relev={qr.relevance_score}/5  "
                f"KW={qr.keyword_recall:.0%}  ${qr.total_cost_usd:.4f}  {qr.latency_s:.1f}s"
            )
            results.append(qr)

        # Aggregate
        answerable = [r for r in results if r.should_answer]
        unanswerable = [r for r in results if not r.should_answer]
        refusal_correct = [r for r in unanswerable if r.correctly_refused is True]

        summary = EvalSummary(
            total_questions=len(results),
            passed=sum(1 for r in results if r.passed()),
            failed=sum(1 for r in results if not r.passed()),
            pass_rate=sum(1 for r in results if r.passed()) / len(results),
            avg_faithfulness=sum(r.faithfulness_score for r in results) / len(results),
            avg_relevance=sum(r.relevance_score for r in results) / len(results),
            avg_keyword_recall=sum(r.keyword_recall for r in results) / len(results),
            refusal_accuracy=len(refusal_correct) / len(unanswerable) if unanswerable else 1.0,
            total_tokens=sum(r.prompt_tokens + r.completion_tokens for r in results),
            total_cost_usd=sum(r.total_cost_usd for r in results),
            avg_latency_s=sum(r.latency_s for r in results) / len(results),
            results=results,
        )

        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: EvalSummary):
        print(f"\n{'='*60}")
        print("  EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Questions        : {summary.total_questions}")
        print(f"  Passed           : {summary.passed} / {summary.total_questions}  ({summary.pass_rate:.0%})")
        print(f"  Avg Faithfulness : {summary.avg_faithfulness:.2f} / 5")
        print(f"  Avg Relevance    : {summary.avg_relevance:.2f} / 5")
        print(f"  Avg Keyword Recall: {summary.avg_keyword_recall:.0%}")
        print(f"  Refusal Accuracy : {summary.refusal_accuracy:.0%}")
        print(f"  Total tokens     : {summary.total_tokens:,}")
        print(f"  Total cost       : ${summary.total_cost_usd:.4f}")
        print(f"  Avg latency      : {summary.avg_latency_s:.1f}s")
        print(f"{'='*60}\n")

        # Category breakdown
        cats = set(r.category for r in summary.results)
        print("  BY CATEGORY:")
        for cat in sorted(cats):
            cr = [r for r in summary.results if r.category == cat]
            cp = sum(1 for r in cr if r.passed())
            print(f"    {cat:<20} {cp}/{len(cr)} passed")
        print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run RAG Agent evaluation")
    parser.add_argument("--questions-file", default="evaluation/eval_questions.json")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not set")
        sys.exit(1)
    # No API key needed for embeddings — local BGE model

    model = os.getenv("GROQ_MODEL") or os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    collection = os.getenv("COLLECTION_NAME", "python_docs")
    top_k = int(os.getenv("TOP_K", "6"))

    from langchain_groq import ChatGroq
    from vectorstore.chroma_store import ChromaVectorStore
    from vectorstore.embeddings import TrackedEmbeddings
    from agent.graph import RAGAgent
    from agent.token_tracker import TokenTracker

    # Build agent
    emb = TrackedEmbeddings()  # local BGE model, no API key
    store = ChromaVectorStore(persist_dir=chroma_dir, collection_name=collection, embeddings=emb)
    tracker = TokenTracker(model=model)
    agent = RAGAgent(
        vector_store=store, model=model, top_k=top_k,
        tracker=tracker, groq_api_key=groq_api_key,
    )

    # Judge LLM (also Groq)
    judge = ChatGroq(model=model, temperature=0, groq_api_key=groq_api_key)

    evaluator = RAGEvaluator(agent=agent, judge_llm=judge, questions_path=args.questions_file)
    summary = evaluator.run()

    if args.output:
        # Serialise results
        output = {
            "summary": {
                "total_questions": summary.total_questions,
                "passed": summary.passed,
                "failed": summary.failed,
                "pass_rate": summary.pass_rate,
                "avg_faithfulness": summary.avg_faithfulness,
                "avg_relevance": summary.avg_relevance,
                "avg_keyword_recall": summary.avg_keyword_recall,
                "refusal_accuracy": summary.refusal_accuracy,
                "total_tokens": summary.total_tokens,
                "total_cost_usd": summary.total_cost_usd,
                "avg_latency_s": summary.avg_latency_s,
            },
            "results": [asdict(r) for r in summary.results],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
