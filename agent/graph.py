"""
agent/graph.py

LangGraph stateful RAG workflow.
Uses Groq for ultra-fast LLM inference (query rewriting, grading, generation).
OpenAI embeddings are used separately in the vector store.

Graph topology:
  START
    │
    ▼
  rewrite_query
    │
    ▼
  retrieve
    │
    ▼
  grade_documents
    │
    ├── (has_relevant_docs=True)  ──▶ generate ──▶ END
    │
    └── (has_relevant_docs=False) ──▶ handle_no_docs ──▶ END
"""

import logging
import os
from functools import partial
from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agent.nodes import generate, grade_documents, handle_no_docs, retrieve, rewrite_query
from agent.token_tracker import QueryUsage, TokenTracker
from vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question: str
    rewritten_query: str
    documents: List[Document]
    answer: str
    sources: List[str]
    has_relevant_docs: bool
    top_k: int
    usage: QueryUsage


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_grading(state: RAGState) -> Literal["generate", "handle_no_docs"]:
    if state.get("has_relevant_docs", False):
        return "generate"
    return "handle_no_docs"


# ---------------------------------------------------------------------------
# RAG Agent
# ---------------------------------------------------------------------------

class RAGAgent:
    """
    High-level agent that wraps the LangGraph workflow.

    Parameters
    ----------
    vector_store  : ChromaVectorStore instance (must be populated)
    model         : OpenAI chat model name
    temperature   : LLM temperature (0 = deterministic)
    top_k         : Number of documents to retrieve
    tracker       : Optional shared TokenTracker for aggregate stats
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0,
        top_k: int = 6,
        tracker: Optional[TokenTracker] = None,
        groq_api_key: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.model = model
        self.top_k = top_k
        self.tracker = tracker or TokenTracker(model=model)

        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
        )

        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        builder = StateGraph(RAGState)

        # Bind LLM/vector_store into each node via partial
        builder.add_node("rewrite_query",   partial(rewrite_query, llm=self.llm))
        builder.add_node("retrieve",        partial(retrieve, vector_store=self.vector_store))
        builder.add_node("grade_documents", partial(grade_documents, llm=self.llm))
        builder.add_node("generate",        partial(generate, llm=self.llm))
        builder.add_node("handle_no_docs",  handle_no_docs)

        # Edges
        builder.add_edge(START, "rewrite_query")
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("retrieve", "grade_documents")
        builder.add_conditional_edges(
            "grade_documents",
            route_after_grading,
            {"generate": "generate", "handle_no_docs": "handle_no_docs"},
        )
        builder.add_edge("generate", END)
        builder.add_edge("handle_no_docs", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Run the RAG pipeline for a single question.
        The Groq LLM is used for query rewriting, document grading, and generation.

        Returns
        -------
        dict with keys:
          - question       : original question
          - rewritten_query: expanded search query
          - answer         : grounded answer string
          - sources        : list of source URLs
          - usage          : QueryUsage object with token/cost stats
        """
        usage = QueryUsage(question=question, model=self.model)

        initial_state: RAGState = {
            "question": question,
            "rewritten_query": "",
            "documents": [],
            "answer": "",
            "sources": [],
            "has_relevant_docs": False,
            "top_k": self.top_k,
            "usage": usage,
        }

        final_state = self._graph.invoke(initial_state)

        # Record in tracker
        self.tracker.record_query(final_state["usage"])

        return {
            "question": final_state["question"],
            "rewritten_query": final_state["rewritten_query"],
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "usage": final_state["usage"],
        }

    def get_graph_diagram(self) -> str:
        """Return ASCII diagram of the LangGraph workflow."""
        try:
            return self._graph.get_graph().draw_ascii()
        except Exception:
            return "(Graph diagram unavailable — install graphviz for ASCII art)"
