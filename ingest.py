"""
ingest.py

Entry point for the ingestion pipeline:
  1. Crawl the target website
  2. Extract, clean, and chunk content
  3. Embed and store in ChromaDB

Usage:
  python ingest.py [--url URL] [--max-pages N] [--reset]
"""

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


def parse_args():
    p = argparse.ArgumentParser(description="Crawl & ingest website into ChromaDB")
    p.add_argument("--url", default=os.getenv("TARGET_URL", "https://docs.python.org/3/"),
                   help="Seed URL to crawl (default: Python docs)")
    p.add_argument("--max-pages", type=int, default=int(os.getenv("MAX_PAGES", "80")),
                   help="Max pages to crawl (default: 80)")
    p.add_argument("--max-depth", type=int, default=int(os.getenv("MAX_DEPTH", "3")),
                   help="Max link depth (default: 3)")
    p.add_argument("--crawl-delay", type=float, default=float(os.getenv("CRAWL_DELAY", "0.5")),
                   help="Seconds between requests (default: 0.5)")
    p.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "800")))
    p.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "150")))
    p.add_argument("--chroma-dir", default=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))
    p.add_argument("--collection", default=os.getenv("COLLECTION_NAME", "python_docs"))
    p.add_argument("--reset", action="store_true",
                   help="Delete existing collection before ingesting")
    return p.parse_args()


def main():
    args = parse_args()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not set — Groq LLM will not work, but ingestion (embeddings) is still fine.")
    # No API key needed for embeddings — Nomic model runs locally.

    print("\n" + "=" * 60)
    print("  RAG AGENT — INGESTION PIPELINE")
    print("=" * 60)
    print(f"  Target URL    : {args.url}")
    print(f"  Max pages     : {args.max_pages}")
    print(f"  Max depth     : {args.max_depth}")
    print(f"  Crawl delay   : {args.crawl_delay}s")
    print(f"  Chunk size    : {args.chunk_size} chars")
    print(f"  Chunk overlap : {args.chunk_overlap} chars")
    print(f"  ChromaDB dir  : {args.chroma_dir}")
    print(f"  Collection    : {args.collection}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Step 1: Crawl
    # ------------------------------------------------------------------
    from crawler.web_crawler import WebCrawler

    logger.info("Step 1/3: Crawling %s ...", args.url)
    t0 = time.time()

    crawler = WebCrawler(
        start_url=args.url,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        crawl_delay=args.crawl_delay,
    )
    pages = crawler.crawl(show_progress=True)

    crawl_time = time.time() - t0
    logger.info("Crawl complete: %d pages in %.1fs", len(pages), crawl_time)

    # ------------------------------------------------------------------
    # Step 2: Process
    # ------------------------------------------------------------------
    from crawler.content_processor import ContentProcessor

    logger.info("Step 2/3: Extracting and chunking content ...")
    t1 = time.time()

    processor = ContentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    docs = processor.process(pages)

    process_time = time.time() - t1
    logger.info("Processing complete: %d chunks in %.1fs", len(docs), process_time)

    if not docs:
        logger.error("No documents produced — check URL and crawl settings.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Embed & store
    # ------------------------------------------------------------------
    from vectorstore.chroma_store import ChromaVectorStore
    from vectorstore.embeddings import TrackedEmbeddings

    logger.info("Step 3/3: Embedding and storing in ChromaDB ...")
    t2 = time.time()

    embeddings = TrackedEmbeddings(model="BAAI/bge-small-en-v1.5")
    store = ChromaVectorStore(
        persist_dir=args.chroma_dir,
        collection_name=args.collection,
        embeddings=embeddings,
    )

    if args.reset:
        logger.warning("--reset flag set: deleting existing collection.")
        store.reset()

    store.add_documents(docs, batch_size=100)

    embed_time = time.time() - t2

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    stats = store.collection_stats()
    total_time = time.time() - t0

    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Pages crawled     : {len(pages)}")
    print(f"  Chunks produced   : {len(docs)}")
    print(f"  Docs in store     : {stats['document_count']}")
    print(f"  Crawl time        : {crawl_time:.1f}s")
    print(f"  Processing time   : {process_time:.1f}s")
    print(f"  Embedding time    : {embed_time:.1f}s")
    print(f"  Total time        : {total_time:.1f}s")
    print()
    print(embeddings.usage_report())
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
