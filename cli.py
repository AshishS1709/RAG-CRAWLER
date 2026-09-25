"""
cli.py

Command-line interface for the RAG agent.

Usage:
  python cli.py "What is a list comprehension in Python?"
  python cli.py --interactive
  python cli.py --show-graph
"""

import argparse
import logging
import os
import sys
from typing import Optional

from colorama import Fore, Style, init
from dotenv import load_dotenv

load_dotenv()
init(autoreset=True)  # colorama

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def print_banner(collection: Optional[str] = None):
    sub = f"Collection: {collection}" if collection else "Grounded answers from knowledge base"
    print(Fore.CYAN + Style.BRIGHT + f"""
╔══════════════════════════════════════════════════════╗
║              Website RAG Agent 🌐                    ║
║   {sub:<51}║
╚══════════════════════════════════════════════════════╝""" + Style.RESET_ALL)


def print_result(result: dict, show_usage: bool = True):
    print()
    print(Fore.YELLOW + "─" * 60 + Style.RESET_ALL)

    print(Fore.GREEN + Style.BRIGHT + "QUESTION:" + Style.RESET_ALL)
    print(f"  {result['question']}")

    if result.get("rewritten_query") and result["rewritten_query"] != result["question"]:
        print(Fore.CYAN + "\nSEARCH QUERY (rewritten):" + Style.RESET_ALL)
        print(f"  {result['rewritten_query']}")

    print(Fore.GREEN + Style.BRIGHT + "\nANSWER:" + Style.RESET_ALL)
    # Indent each line for readability
    for line in result["answer"].splitlines():
        print(f"  {line}")

    if result.get("sources"):
        print(Fore.BLUE + "\nSOURCES:" + Style.RESET_ALL)
        for url in result["sources"]:
            print(f"  • {url}")

    if show_usage and result.get("usage"):
        u = result["usage"]
        print(Fore.MAGENTA + "\nTOKEN USAGE:" + Style.RESET_ALL)
        print(u.summary_table())

    print(Fore.YELLOW + "─" * 60 + Style.RESET_ALL)
    print()


def build_agent(collection_name: Optional[str] = None, model: Optional[str] = None):
    """Initialise and return a RAGAgent."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print(Fore.RED + "Error: GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys" + Style.RESET_ALL)
        sys.exit(1)
    # Embeddings run locally — no API key needed

    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    collection = collection_name or os.getenv("COLLECTION_NAME", "python_docs")
    model_name = model or os.getenv("GROQ_MODEL") or os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
    top_k = int(os.getenv("TOP_K", "6"))

    from vectorstore.chroma_store import ChromaVectorStore
    from vectorstore.embeddings import TrackedEmbeddings
    from agent.graph import RAGAgent
    from agent.token_tracker import TokenTracker

    print(Fore.CYAN + "Loading vector store...", end=" ", flush=True)
    embeddings = TrackedEmbeddings(model="BAAI/bge-small-en-v1.5")
    store = ChromaVectorStore(
        persist_dir=chroma_dir,
        collection_name=collection,
        embeddings=embeddings,
    )
    stats = store.collection_stats()
    print(Fore.GREEN + f"OK ({stats['document_count']:,} chunks loaded)" + Style.RESET_ALL)

    if stats["document_count"] == 0:
        print(Fore.RED + "\nWarning: Vector store is empty! Run `python ingest.py` first." + Style.RESET_ALL)
        sys.exit(1)

    tracker = TokenTracker(model=model_name)
    agent = RAGAgent(
        vector_store=store,
        model=model_name,
        top_k=top_k,
        tracker=tracker,
        groq_api_key=groq_api_key,
    )
    return agent


def parse_args():
    p = argparse.ArgumentParser(description="Python Docs RAG Agent CLI")
    p.add_argument("question", nargs="?", help="Question to ask the agent")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="Start interactive REPL mode")
    p.add_argument("--collection", default=None,
                   help="ChromaDB collection name (defaults to COLLECTION_NAME env or 'python_docs')")
    p.add_argument("--model", default=None,
                   help="Groq chat model name (defaults to GROQ_MODEL / CHAT_MODEL env or 'openai/gpt-oss-20b')")
    p.add_argument("--no-usage", action="store_true",
                   help="Hide token usage stats")
    p.add_argument("--show-graph", action="store_true",
                   help="Print the LangGraph workflow diagram and exit")
    return p.parse_args()


def main():
    args = parse_args()
    target_coll = args.collection or os.getenv("COLLECTION_NAME", "python_docs")
    print_banner(collection=target_coll)

    agent = build_agent(collection_name=args.collection, model=args.model)

    if args.show_graph:
        print(Fore.CYAN + "\nLangGraph Workflow:\n" + Style.RESET_ALL)
        print(agent.get_graph_diagram())
        return

    if args.question:
        # Single question mode
        result = agent.ask(args.question)
        print_result(result, show_usage=not args.no_usage)
        print(agent.tracker.summary_report())

    elif args.interactive:
        # Interactive REPL
        print(Fore.CYAN + "Interactive mode. Type 'quit' or 'exit' to stop.\n" + Style.RESET_ALL)
        while True:
            try:
                question = input(Fore.WHITE + Style.BRIGHT + "You: " + Style.RESET_ALL).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break

            result = agent.ask(question)
            print_result(result, show_usage=not args.no_usage)

        print(agent.tracker.summary_report())
    else:
        print(Fore.YELLOW + "No question provided. Use --help for usage." + Style.RESET_ALL)
        print("  Example: python cli.py \"What is a list comprehension?\"")
        print("  Example: python cli.py --interactive")


if __name__ == "__main__":
    main()
