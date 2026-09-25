"""
agent/nodes.py

Individual node functions for the LangGraph RAG workflow.

Nodes:
  rewrite_query    → expand/clarify the user question
  retrieve         → semantic search the vector store
  grade_documents  → filter irrelevant retrieved docs
  generate         → grounded answer using LLM
  handle_no_docs   → respond when no relevant docs found
"""

import json
import logging
from typing import Any, Dict, List

import tiktoken
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from agent.prompts import (
    GENERATE_HUMAN,
    GENERATE_SYSTEM,
    GRADE_DOCUMENT_HUMAN,
    GRADE_DOCUMENT_SYSTEM,
    NO_DOCS_RESPONSE,
    QUERY_REWRITE_HUMAN,
    QUERY_REWRITE_SYSTEM,
)
from agent.token_tracker import QueryUsage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared token counter
# ---------------------------------------------------------------------------

def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Node: rewrite_query
# ---------------------------------------------------------------------------

def rewrite_query(state: Dict[str, Any], llm: BaseChatModel) -> Dict[str, Any]:
    """
    Rewrite the user's question into a clearer retrieval query.
    Tracks prompt/completion tokens consumed.
    """
    question = state["question"]
    usage: QueryUsage = state["usage"]

    messages = [
        SystemMessage(content=QUERY_REWRITE_SYSTEM),
        HumanMessage(content=QUERY_REWRITE_HUMAN.format(question=question)),
    ]

    # Track input tokens
    prompt_text = QUERY_REWRITE_SYSTEM + QUERY_REWRITE_HUMAN.format(question=question)
    usage.prompt_tokens += _count_tokens(prompt_text, usage.model)

    response = llm.invoke(messages)
    rewritten = response.content.strip()

    # Track output tokens
    usage.completion_tokens += _count_tokens(rewritten, usage.model)

    logger.debug("Query rewritten: %r → %r", question, rewritten)

    return {**state, "rewritten_query": rewritten}


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------

def retrieve(state: Dict[str, Any], vector_store) -> Dict[str, Any]:
    """
    Retrieve top-k documents from the vector store.
    Also tracks embedding tokens for the query.
    """
    import tiktoken as _tiktoken

    query = state.get("rewritten_query") or state["question"]
    usage: QueryUsage = state["usage"]

    # Count embedding tokens
    try:
        enc = _tiktoken.encoding_for_model("text-embedding-3-small")
    except KeyError:
        enc = _tiktoken.get_encoding("cl100k_base")
    usage.embedding_tokens += len(enc.encode(query))

    docs = vector_store.similarity_search(query, k=state.get("top_k", 6))
    logger.debug("Retrieved %d docs for query: %r", len(docs), query[:60])

    return {**state, "documents": docs}


# ---------------------------------------------------------------------------
# Node: grade_documents
# ---------------------------------------------------------------------------

def grade_documents(state: Dict[str, Any], llm: BaseChatModel) -> Dict[str, Any]:
    """
    Score each retrieved document for relevance to the user question.
    Filters out irrelevant documents and sets 'has_relevant_docs'.
    """
    question = state["question"]
    documents: List[Document] = state["documents"]
    usage: QueryUsage = state["usage"]

    relevant_docs = []

    for doc in documents:
        prompt_text = GRADE_DOCUMENT_SYSTEM + GRADE_DOCUMENT_HUMAN.format(
            question=question,
            document=doc.page_content[:1500],
        )
        usage.prompt_tokens += _count_tokens(prompt_text, usage.model)

        messages = [
            SystemMessage(content=GRADE_DOCUMENT_SYSTEM),
            HumanMessage(content=GRADE_DOCUMENT_HUMAN.format(
                question=question,
                document=doc.page_content[:1500],
            )),
        ]
        try:
            response = llm.invoke(messages)
            completion_text = response.content
            result = json.loads(response.content.strip())
            score = result.get("score", "no").lower()
        except Exception as e:
            logger.warning("Grading parse error: %s — defaulting to 'yes'", e)
            score = "yes"
            completion_text = "yes"

        usage.completion_tokens += _count_tokens(completion_text, usage.model)

        if score == "yes":
            relevant_docs.append(doc)
            logger.debug("✓ Relevant: %s", doc.metadata.get("url", ""))
        else:
            logger.debug("✗ Filtered: %s", doc.metadata.get("url", ""))

    has_relevant = len(relevant_docs) > 0
    logger.info("Grading: %d/%d docs deemed relevant", len(relevant_docs), len(documents))

    return {**state, "documents": relevant_docs, "has_relevant_docs": has_relevant}


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------

def _build_context(docs: List[Document]) -> str:
    """Format documents into a numbered context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        url = doc.metadata.get("url", "unknown")
        title = doc.metadata.get("title", "")
        parts.append(f"[{i}] Source: {url}\nTitle: {title}\n\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_sources(docs: List[Document]) -> List[str]:
    """Deduplicate and return source URLs, preserving order."""
    seen = set()
    sources = []
    for doc in docs:
        url = doc.metadata.get("url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    return sources


def generate(state: Dict[str, Any], llm: BaseChatModel) -> Dict[str, Any]:
    """
    Generate a grounded answer from relevant documents.
    """
    question = state["question"]
    docs: List[Document] = state["documents"]
    usage: QueryUsage = state["usage"]

    context = _build_context(docs)
    system_prompt = GENERATE_SYSTEM.format(context=context)
    human_prompt = GENERATE_HUMAN.format(question=question)

    prompt_text = system_prompt + human_prompt
    usage.prompt_tokens += _count_tokens(prompt_text, usage.model)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    usage.completion_tokens += _count_tokens(answer, usage.model)

    sources = _extract_sources(docs)

    logger.info("Answer generated (%d chars, %d sources)", len(answer), len(sources))
    return {**state, "answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Node: handle_no_docs
# ---------------------------------------------------------------------------

def handle_no_docs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a graceful 'not enough info' response when no relevant docs found."""
    return {**state, "answer": NO_DOCS_RESPONSE, "sources": []}
