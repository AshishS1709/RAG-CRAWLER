"""
vectorstore/embeddings.py

Local HuggingFace embeddings using nomic-ai/nomic-embed-text-v1.5.
No API key required — model runs entirely on your machine.

Model: nomic-ai/nomic-embed-text-v1.5
Source: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
Cost: $0.00 (local inference)
Dimensions: 768

First run will download ~274 MB model weights from HuggingFace Hub.
Subsequent runs load from the local HuggingFace cache.

Nomic task-type prefixes (important for best retrieval quality):
  "search_document: <text>"  → when embedding corpus chunks (ingestion)
  "search_query: <text>"     → when embedding user queries (retrieval)
"""

import logging
from typing import List, Optional, Type

import tiktoken
import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

# Nomic task-type prefixes
TASK_SEARCH_DOC   = "search_document"
TASK_SEARCH_QUERY = "search_query"


# ---------------------------------------------------------------------------
# Core HuggingFace helpers
# ---------------------------------------------------------------------------

def _mean_pooling(model_output, attention_mask) -> torch.Tensor:
    """Mean-pool token embeddings, weighted by attention mask."""
    token_embeddings = model_output[0]  # shape: (batch, seq_len, hidden)
    mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
        mask_expanded.sum(1), min=1e-9
    )


def _add_task_prefix(texts: List[str], task: str) -> List[str]:
    """Prepend Nomic task-type prefix to each text."""
    return [f"{task}: {text}" for text in texts]


# ---------------------------------------------------------------------------
# LangChain-compatible Embeddings class
# ---------------------------------------------------------------------------

class NomicLocalEmbeddings(Embeddings):
    """
    LangChain Embeddings backed by the locally loaded nomic-embed-text-v1.5 model.
    Implements the Embeddings interface required by ChromaDB + LangChain.

    Parameters
    ----------
    model_name : HuggingFace model ID (default: nomic-ai/nomic-embed-text-v1.5)
    device     : 'auto', 'cpu', 'cuda', or 'mps'  (default: auto-detect)
    batch_size : Number of texts to encode per forward pass
    max_length : Max token length (Nomic supports up to 8192)
    """

    def __init__(
        self,
        model_name: str = MODEL_ID,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

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

        logger.info("Loading Nomic model '%s' on device='%s'...", model_name, self.device)

        # Load tokenizer and model (exactly as provided by the user)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=self.device if self.device != "cpu" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()
        logger.info("Nomic model loaded ✓ (device=%s)", self.device)

    # ------------------------------------------------------------------
    # Internal: encode a batch of strings
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """Tokenize → forward pass → mean-pool → L2-normalise."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            # Move tensors to correct device
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self.model(**encoded)

            pooled = _mean_pooling(output, encoded["attention_mask"])
            normalised = F.normalize(pooled, p=2, dim=1)

            all_embeddings.extend(normalised.cpu().float().tolist())

        return all_embeddings

    # ------------------------------------------------------------------
    # LangChain Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed corpus documents with 'search_document' task prefix."""
        prefixed = _add_task_prefix(texts, TASK_SEARCH_DOC)
        logger.debug("Embedding %d documents (task=search_document)", len(texts))
        return self._encode(prefixed)

    def embed_query(self, text: str) -> List[float]:
        """Embed a user query with 'search_query' task prefix."""
        prefixed = _add_task_prefix([text], TASK_SEARCH_QUERY)
        logger.debug("Embedding query (task=search_query): %r", text[:60])
        return self._encode(prefixed)[0]


# ---------------------------------------------------------------------------
# TrackedEmbeddings — wraps NomicLocalEmbeddings with token counting
# ---------------------------------------------------------------------------

class TrackedEmbeddings:
    """
    Wraps NomicLocalEmbeddings with cumulative token/cost tracking.
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
        # Kept for API compatibility — not used (no remote API)
        api_key: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 32,
    ):
        self.model = model
        self._client = NomicLocalEmbeddings(
            model_name=model,
            device=device,
            batch_size=batch_size,
        )
        # Use cl100k_base as approximate tokenizer for counting
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._total_tokens: int = 0
        self._total_calls: int = 0

    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._total_tokens += sum(len(self._enc.encode(t)) for t in texts)
        self._total_calls += 1
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        self._total_tokens += len(self._enc.encode(text))
        self._total_calls += 1
        return self._client.embed_query(text)

    @property
    def langchain_embeddings(self) -> NomicLocalEmbeddings:
        """Return the underlying LangChain-compatible embeddings (for ChromaDB)."""
        return self._client

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return 0.0  # Local inference — always free

    def usage_report(self) -> str:
        return (
            f"Embedding model : {self.model}\n"
            f"Backend         : HuggingFace Transformers (local)\n"
            f"Device          : {self._client.device}\n"
            f"Total tokens    : {self._total_tokens:,}\n"
            f"Total API calls : {self._total_calls:,}\n"
            f"Estimated cost  : $0.0000 (local — FREE 🎉)"
        )
