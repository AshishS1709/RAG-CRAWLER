"""
crawler/content_processor.py

Takes raw CrawledPage objects and produces clean, chunked
LangChain Documents ready for embedding.

Pipeline:
  1. Extract readable text with trafilatura (falls back to BS4)
  2. Clean & normalise whitespace
  3. Split into overlapping chunks via RecursiveCharacterTextSplitter
  4. Attach metadata: url, title, chunk_index, total_chunks
"""

import logging
import re
from typing import List, Optional

import trafilatura
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from crawler.web_crawler import CrawledPage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_with_trafilatura(html: str, url: str) -> Optional[str]:
    """Use trafilatura for high-quality article extraction."""
    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=False,
        )
        return text
    except Exception as exc:
        logger.debug("trafilatura failed for %s: %s", url, exc)
        return None


def _extract_with_bs4(html: str) -> str:
    """
    Fallback: Generic BeautifulSoup content extraction.
    Decomposes boilerplate tags (nav, header, footer, aside, scripts, forms)
    and extracts from semantic content containers (main, article, [role="main"])
    or the largest remaining text block.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove boilerplate and non-content tags
    for tag in soup.select(
        "nav, header, footer, aside, script, style, noscript, form, svg, "
        ".sidebar, .nav, .header, .footer, .menu, .ad, .advertisement, "
        ".cookie-banner, .headerlink, .sphinxsidebar, #indices-and-tables, "
        ".related, .sphinxsidebarwrapper"
    ):
        tag.decompose()

    # Priority 1: Semantic main content containers
    for selector in ["main", "article", "[role='main']", "#main-content", "#content", ".main-content", ".content", "div.body"]:
        container = soup.select_one(selector)
        if container:
            text = container.get_text(separator="\n", strip=True)
            if len(text) > 100:
                return text

    # Priority 2: Largest remaining text block
    body = soup.find("body") or soup
    candidates = body.find_all(["div", "section", "article"], recursive=False)
    if candidates:
        best_block = max(candidates, key=lambda el: len(el.get_text(strip=True)))
        best_text = best_block.get_text(separator="\n", strip=True)
        if len(best_text) > 100:
            return best_text

    return body.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise whitespace, remove excessive blank lines."""
    # Collapse multiple spaces (but not newlines)
    text = re.sub(r" {2,}", " ", text)
    # Collapse 3+ consecutive newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    # Remove lines that are purely punctuation/symbols (navigation artifacts)
    lines = [l for l in lines if len(l) > 2 or l == ""]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Content Processor
# ---------------------------------------------------------------------------

class ContentProcessor:
    """
    Converts CrawledPage objects → list[Document] (chunked).

    Parameters
    ----------
    chunk_size    : Target token/character size per chunk
    chunk_overlap : Overlap between consecutive chunks
    min_text_len  : Pages with fewer characters are discarded
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_text_len: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_text_len = min_text_len

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pages: List[CrawledPage]) -> List[Document]:
        """
        Process a list of crawled pages into chunked Documents.

        Returns a flat list of Documents across all pages.
        """
        all_docs: List[Document] = []
        skipped = 0

        for page in pages:
            docs = self._process_page(page)
            if docs is None:
                skipped += 1
            else:
                all_docs.extend(docs)

        logger.info(
            "Content processing complete: %d chunks from %d pages (%d skipped)",
            len(all_docs),
            len(pages) - skipped,
            skipped,
        )
        return all_docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_page(self, page: CrawledPage) -> Optional[List[Document]]:
        """Extract, clean, and chunk a single page. Returns None if skipped."""

        # 1. Extract text
        text = _extract_with_trafilatura(page.raw_html, page.url)
        if not text or len(text) < self.min_text_len:
            text = _extract_with_bs4(page.raw_html)

        # 2. Clean
        text = _clean_text(text)

        # 3. Skip if still too short
        if len(text) < self.min_text_len:
            logger.debug("Skipping short page: %s (%d chars)", page.url, len(text))
            return None

        # 4. Split into chunks
        raw_chunks = self.splitter.split_text(text)
        if not raw_chunks:
            return None

        # 5. Build Documents with rich metadata
        docs = []
        for i, chunk in enumerate(raw_chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "url": page.url,
                    "title": page.title,
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "depth": page.depth,
                    "char_count": len(chunk),
                },
            ))

        logger.debug(
            "Page processed: %s → %d chunks (title=%r)",
            page.url, len(docs), page.title[:60],
        )
        return docs
