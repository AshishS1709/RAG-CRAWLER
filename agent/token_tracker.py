"""
agent/token_tracker.py

Per-query and cumulative token + cost tracking for OpenAI models.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# Pricing (USD per 1M tokens, as of 2024-Q4)
# ── Groq models (chat/generation) ──────────────────────────────────────
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Groq — extremely fast inference
    "llama-3.1-8b-instant":    {"input": 0.05,  "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59,  "output": 0.79},
    "llama-3.1-70b-versatile": {"input": 0.59,  "output": 0.79},
    "mixtral-8x7b-32768":      {"input": 0.24,  "output": 0.24},
    "gemma2-9b-it":            {"input": 0.20,  "output": 0.20},
    # OpenAI (kept for reference if switched back)
    "gpt-4o-mini":   {"input": 0.150, "output": 0.600},
    "gpt-4o":        {"input": 5.000, "output": 15.000},
    "gpt-3.5-turbo": {"input": 0.500, "output": 1.500},
}


@dataclass
class QueryUsage:
    """Token usage and cost for a single query."""
    question: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    model: str = "gpt-4o-mini"

    @property
    def total_llm_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def llm_cost_usd(self) -> float:
        pricing = MODEL_PRICING.get(self.model, {"input": 0.150, "output": 0.600})
        return (
            self.prompt_tokens / 1_000_000 * pricing["input"]
            + self.completion_tokens / 1_000_000 * pricing["output"]
        )

    @property
    def embedding_cost_usd(self) -> float:
        # Local embedding model (BGE) — $0 for embeddings
        return 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.llm_cost_usd + self.embedding_cost_usd

    def as_dict(self) -> dict:
        return {
            "question": self.question[:80],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_llm_tokens": self.total_llm_tokens,
            "llm_cost_usd": round(self.llm_cost_usd, 6),
            "embedding_cost_usd": round(self.embedding_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
        }

    def summary_table(self) -> str:
        rows = [
            ("Prompt tokens",      f"{self.prompt_tokens:>10,}"),
            ("Completion tokens",  f"{self.completion_tokens:>10,}"),
            ("Embedding tokens",   f"{self.embedding_tokens:>10,}"),
            ("Total LLM tokens",   f"{self.total_llm_tokens:>10,}"),
            ("LLM cost",           f"${self.llm_cost_usd:>10.5f}"),
            ("Embedding cost",     f"${self.embedding_cost_usd:>10.5f}"),
            ("Total cost",         f"${self.total_cost_usd:>10.5f}"),
        ]
        lines = [f"  {'Metric':<22} {'Value':>12}"]
        lines.append("  " + "-" * 36)
        for label, val in rows:
            lines.append(f"  {label:<22} {val:>12}")
        return "\n".join(lines)


@dataclass
class TokenTracker:
    """Accumulates usage across multiple queries for aggregate reporting."""

    model: str = "gpt-4o-mini"
    history: List[QueryUsage] = field(default_factory=list)

    # Ingestion (embedding-only)
    ingestion_tokens: int = 0
    ingestion_cost_usd: float = 0.0

    def record_ingestion(self, tokens: int) -> None:
        self.ingestion_tokens = tokens
        self.ingestion_cost_usd = tokens / 1_000_000 * 0.020

    def record_query(self, usage: QueryUsage) -> None:
        self.history.append(usage)

    # ------------------------------------------------------------------
    # Aggregate stats
    # ------------------------------------------------------------------

    @property
    def total_query_llm_tokens(self) -> int:
        return sum(u.total_llm_tokens for u in self.history)

    @property
    def total_query_embedding_tokens(self) -> int:
        return sum(u.embedding_tokens for u in self.history)

    @property
    def total_query_cost_usd(self) -> float:
        return sum(u.total_cost_usd for u in self.history)

    @property
    def total_cost_usd(self) -> float:
        return self.ingestion_cost_usd + self.total_query_cost_usd

    def projection(self, n_queries: int) -> dict:
        """Estimate costs at scale given average per-query cost."""
        if not self.history:
            avg_per_query = 0.003  # conservative fallback
        else:
            avg_per_query = self.total_query_cost_usd / len(self.history)
        return {
            "avg_cost_per_query": avg_per_query,
            f"estimated_cost_{n_queries}_queries": self.ingestion_cost_usd + avg_per_query * n_queries,
        }

    def summary_report(self) -> str:
        lines = [
            "\n" + "=" * 50,
            "  TOKEN USAGE & COST SUMMARY",
            "=" * 50,
            f"  Model             : {self.model}",
            f"  Queries answered  : {len(self.history)}",
            "",
            "  INGESTION (one-time)",
            f"    Embedding tokens: {self.ingestion_tokens:,}",
            f"    Ingestion cost  : ${self.ingestion_cost_usd:.4f}",
            "",
            "  QUERIES (cumulative)",
            f"    LLM tokens      : {self.total_query_llm_tokens:,}",
            f"    Embedding tokens: {self.total_query_embedding_tokens:,}",
            f"    Query cost      : ${self.total_query_cost_usd:.4f}",
            "",
            "  TOTAL COST        : ${:.4f}".format(self.total_cost_usd),
            "",
            "  COST PROJECTIONS",
        ]
        for n in [100, 1_000, 10_000]:
            proj = self.projection(n)
            lines.append(
                f"    @ {n:>6,} queries: ${proj[f'estimated_cost_{n}_queries']:.2f}"
            )
        lines.append("=" * 50)
        return "\n".join(lines)
