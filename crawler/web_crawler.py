"""
crawler/web_crawler.py

Async BFS web crawler scoped to a single domain.
Respects robots.txt, configurable depth/page limits,
and polite crawl delay between requests.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CrawledPage:
    url: str
    title: str
    raw_html: str
    status_code: int
    depth: int
    links_found: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Robots.txt helper
# ---------------------------------------------------------------------------

class RobotsChecker:
    """Cache-backed robots.txt checker."""

    def __init__(self, user_agent: str = "RAGCrawler/1.0"):
        self.user_agent = user_agent
        self._cache: dict[str, RobotFileParser] = {}

    def _get_parser(self, base_url: str) -> RobotFileParser:
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        if robots_url not in self._cache:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                # If robots.txt is inaccessible, allow everything
                pass
            self._cache[robots_url] = rp
        return self._cache[robots_url]

    def can_fetch(self, url: str) -> bool:
        try:
            parser = self._get_parser(url)
            return parser.can_fetch(self.user_agent, url)
        except Exception:
            return True  # Allow on error


# ---------------------------------------------------------------------------
# Core Crawler
# ---------------------------------------------------------------------------

class WebCrawler:
    """
    BFS crawler that stays within a single domain.

    Parameters
    ----------
    start_url   : Seed URL (e.g. "https://docs.python.org/3/")
    max_pages   : Hard cap on total pages fetched
    max_depth   : Maximum link depth from the seed
    crawl_delay : Seconds to wait between HTTP requests
    url_filter  : Optional callable; return True to include a URL
    user_agent  : User-Agent header sent with every request
    """

    DEFAULT_HEADERS = {
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # URL path prefixes to skip for Python docs (non-content pages)
    SKIP_PATTERNS = [
        "/_sources/",
        "/genindex",
        "/py-modindex",
        "/search",
        "/_static/",
        "/_images/",
        "/download",
        ".zip",
        ".tar",
        ".pdf",
        ".epub",
        "#",           # fragment-only
    ]

    def __init__(
        self,
        start_url: str,
        max_pages: int = 80,
        max_depth: int = 3,
        crawl_delay: float = 0.5,
        url_filter=None,
        user_agent: str = "RAGCrawler/1.0 (assessment project)",
    ):
        self.start_url = start_url.rstrip("/")
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.crawl_delay = crawl_delay
        self.url_filter = url_filter
        self.user_agent = user_agent

        parsed = urlparse(start_url)
        self.base_domain = parsed.netloc
        # Keep trailing slash so "/3/" only matches "/3/..." not "/3.12/..."
        self.base_path_prefix = parsed.path if parsed.path.endswith("/") else parsed.path + "/"

        self.session = requests.Session()
        self.session.headers.update({
            **self.DEFAULT_HEADERS,
            "User-Agent": self.user_agent,
        })

        self.robots = RobotsChecker(user_agent=self.user_agent)
        self.visited: Set[str] = set()
        self.pages: List[CrawledPage] = []

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def _normalise(self, url: str) -> str:
        """Strip fragment, trailing slash, normalise scheme."""
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}{p.path.rstrip('/')}{'?' + p.query if p.query else ''}"

    def _is_in_scope(self, url: str) -> bool:
        """True if the URL belongs to the target domain & path prefix."""
        p = urlparse(url)
        if p.netloc != self.base_domain:
            return False
        if not (p.path.startswith(self.base_path_prefix) or p.path.rstrip("/") == self.base_path_prefix.rstrip("/")):
            return False
        for pat in self.SKIP_PATTERNS:
            if pat in url:
                return False
        if self.url_filter and not self.url_filter(url):
            return False
        return True

    def _extract_links(self, page_url: str, html: str) -> List[str]:
        soup = BeautifulSoup(html, "lxml")
        links = []
        for tag in soup.find_all("a", href=True):
            absolute = urljoin(page_url, tag["href"])
            normalised = self._normalise(absolute)
            if self._is_in_scope(normalised) and normalised not in self.visited:
                links.append(normalised)
        return list(dict.fromkeys(links))  # deduplicate, preserve order

    def _extract_title(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        tag = soup.find("title")
        return tag.get_text(strip=True) if tag else "Untitled"

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> Optional[requests.Response]:
        try:
            resp = self.session.get(url, timeout=15, allow_redirects=True)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None
            return resp
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # BFS crawl
    # ------------------------------------------------------------------

    def crawl(self, show_progress: bool = True) -> List[CrawledPage]:
        """
        Run BFS from start_url.  Returns list of CrawledPage objects.
        """
        from tqdm import tqdm

        queue: deque[tuple[str, int]] = deque()
        queue.append((self._normalise(self.start_url), 0))
        self.visited.add(self._normalise(self.start_url))

        pbar = tqdm(total=self.max_pages, desc="Crawling", unit="page") if show_progress else None

        while queue and len(self.pages) < self.max_pages:
            url, depth = queue.popleft()

            # Robots check
            if not self.robots.can_fetch(url):
                logger.debug("robots.txt blocks: %s", url)
                continue

            resp = self._fetch(url)
            if resp is None:
                continue

            html = resp.text
            title = self._extract_title(html)
            links = self._extract_links(url, html) if depth < self.max_depth else []

            page = CrawledPage(
                url=url,
                title=title,
                raw_html=html,
                status_code=resp.status_code,
                depth=depth,
                links_found=links,
            )
            self.pages.append(page)

            if show_progress and pbar:
                pbar.update(1)
                pbar.set_postfix({"depth": depth, "url": url[-60:]})

            logger.info("[%d/%d] depth=%d  %s", len(self.pages), self.max_pages, depth, url)

            # Enqueue discovered links
            for link in links:
                if link not in self.visited and len(self.visited) < self.max_pages * 3:
                    self.visited.add(link)
                    queue.append((link, depth + 1))

            # Polite delay
            if self.crawl_delay > 0:
                time.sleep(self.crawl_delay)

        if pbar:
            pbar.close()

        logger.info("Crawl complete. %d pages fetched.", len(self.pages))
        return self.pages
