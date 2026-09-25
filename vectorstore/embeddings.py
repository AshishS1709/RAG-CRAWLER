"""
vectorstore/embeddings.py

Local embeddings using BAAI/bge-small-en-v1.5 via SentenceTransformers.
No API key required — model runs entirely on your machine.

Model: BAAI/bge-small-en-v1.5
Source: https://huggingface.co/BAAI/bge-small-en-v1.5
Cost: $0.00 (local inference)
Dimensions: 384

First run will download ~133 MB model weights from HuggingFace Hub.
Subsequent runs load from the local HuggingFace cache.

BGE retrieval instructions:
  - Documents: embed raw text
  - Queries: prepend "Represent this sentence for searching relevant passages: "
"""

import logging
from typing import List, Optional

import tiktoken
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_ID = "BAAI/bge-small-en-v1.5"

# BGE instruction prefix recommended for retrieval queries
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGELocalEmbeddings(Embeddings):
    """
    LangChain Embeddings backed by BAAI/bge-small-en-v1.5 via SentenceTransformers.
    Implements the Embeddings interface required by ChromaDB + LangChain.
    """

    def __init__(
        self,
        model_name: str = MODEL_ID,
        device: str = "auto",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info("Loading BGE model '%s' on device='%s'...", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info("BGE model loaded ✓ (device=%s)", self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed corpus documents with normalized vectors."""
        logger.debug("Embedding %d documents with BGE", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a user query with BGE recommended prefix."""
        prefixed = f"{BGE_QUERY_PREFIX}{text}"
        logger.debug("Embedding query (BGE prefix applied): %r", text[:60])
        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


# Backwards compatibility alias
NomicLocalEmbeddings = BGELocalEmbeddings


class TrackedEmbeddings:
    """
    Wraps BGELocalEmbeddings with cumulative token/cost tracking.
    Cost is always $0 (local inference).

    Usage
    -----
    emb = TrackedEmbeddings()
    vectors = emb.embed_documents(texts)
    print(emb.usage_report())
    """

    def __init__(
        self,
        model: str = MODEL_ID,
        api_key: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 64,
    ):
        self.model = model
        self._client = BGELocalEmbeddings(
            model_name=model,
            device=device,
            batch_size=batch_size,
        )
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._total_tokens: int = 0
        self._total_calls: int = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._total_tokens += sum(len(self._enc.encode(t)) for t in texts)
        self._total_calls += 1
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        self._total_tokens += len(self._enc.encode(text))
        self._total_calls += 1
        return self._client.embed_query(text)

    @property
    def langchain_embeddings(self) -> BGELocalEmbeddings:
        """Return the underlying LangChain-compatible embeddings (for ChromaDB)."""
        return self._client

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return 0.0  # Local inference — always free

    def usage_report(self) -> str:
        return (
            f"Embedding model : {self.model}\n"
            f"Backend         : SentenceTransformers (local)\n"
            f"Device          : {self._client.device}\n"
            f"Total tokens    : {self._total_tokens:,}\n"
            f"Total API calls : {self._total_calls:,}\n"
            f"Estimated cost  : $0.0000 (local — FREE 🎉)"
        )
