"""
vectorstore/chroma_store.py

ChromaDB-backed vector store with persistence.
Provides add, retrieve, and collection-stats helpers.
"""

import logging
import os
from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from vectorstore.embeddings import TrackedEmbeddings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Wrapper around LangChain's Chroma integration with persistence.

    Parameters
    ----------
    persist_dir     : Directory for on-disk ChromaDB storage
    collection_name : Name of the ChromaDB collection
    embeddings      : TrackedEmbeddings instance
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "python_docs",
        embeddings: Optional[TrackedEmbeddings] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embeddings = embeddings or TrackedEmbeddings()

        os.makedirs(persist_dir, exist_ok=True)

        self._chroma_client = chromadb.PersistentClient(path=persist_dir)
        self._store: Optional[Chroma] = None

    # ------------------------------------------------------------------
    # Initialise / load store
    # ------------------------------------------------------------------

    def _get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                client=self._chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings.langchain_embeddings,
            )
        return self._store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, docs: List[Document], batch_size: int = 100) -> int:
        """
        Add documents to the store in batches.
        Returns number of documents added.
        """
        store = self._get_store()
        total = len(docs)
        added = 0

        for i in range(0, total, batch_size):
            batch = docs[i : i + batch_size]
            store.add_documents(batch)
            added += len(batch)
            logger.info(
                "Stored batch %d/%d (%d docs)",
                i // batch_size + 1,
                (total + batch_size - 1) // batch_size,
                len(batch),
            )

        logger.info("Added %d documents to collection '%s'", added, self.collection_name)
        return added

    def similarity_search(
        self,
        query: str,
        k: int = 6,
        filter_dict: Optional[dict] = None,
    ) -> List[Document]:
        """Return top-k documents most similar to query."""
        store = self._get_store()
        return store.similarity_search(query, k=k, filter=filter_dict)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
    ) -> List[tuple[Document, float]]:
        """Return (Document, score) pairs; lower score = more similar."""
        store = self._get_store()
        return store.similarity_search_with_score(query, k=k)

    def as_retriever(self, k: int = 6):
        """Return a LangChain-compatible retriever."""
        store = self._get_store()
        return store.as_retriever(search_kwargs={"k": k})

    def collection_stats(self) -> dict:
        """Return basic stats about the collection."""
        domains = []
        try:
            col = self._chroma_client.get_collection(self.collection_name)
            count = col.count()
            if count > 0:
                sample = col.get(limit=10, include=["metadatas"])
                found_domains = set()
                for meta in sample.get("metadatas", []):
                    if meta:
                        url = meta.get("url") or meta.get("source") or ""
                        if url.startswith("http"):
                            from urllib.parse import urlparse
                            netloc = urlparse(url).netloc
                            if netloc:
                                found_domains.add(netloc)
                domains = sorted(list(found_domains))
        except Exception:
            count = 0
        return {
            "collection": self.collection_name,
            "persist_dir": self.persist_dir,
            "document_count": count,
            "domains": domains,
        }

    def reset(self) -> None:
        """Delete and recreate the collection (destructive!)."""
        try:
            self._chroma_client.delete_collection(self.collection_name)
            logger.warning("Deleted collection '%s'", self.collection_name)
        except Exception:
            pass
        self._store = None
