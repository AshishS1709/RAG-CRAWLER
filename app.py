"""
app.py

Streamlit web UI for the Website-Grounded RAG Agent.

Launch with:
  streamlit run app.py
"""

import os
import re
import sys
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Website RAG Agent",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background: #0f1117; }

  /* Chat messages */
  .user-msg {
    background: linear-gradient(135deg, #1e3a5f, #1a2e4a);
    border-left: 4px solid #4a9eff;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
  }
  .assistant-msg {
    background: linear-gradient(135deg, #1a2e1a, #1e3a1e);
    border-left: 4px solid #4aff6a;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
  }

  /* Source pills */
  .source-pill {
    display: inline-block;
    background: #1e2a3e;
    border: 1px solid #4a9eff44;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78em;
    margin: 3px 4px;
    color: #7ab8ff;
  }

  /* Sidebar metrics */
  .metric-card {
    background: #1a1f2e;
    border: 1px solid #2a3048;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
  }

  /* Header */
  h1 { color: #e8f4ff !important; }
  .subtitle { color: #7a9abf; font-size: 1.0em; margin-top: -12px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Your Groq API key. Get one free at console.groq.com/keys",
    )
    available_models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "mixtral-8x7b-32768",
    ]
    env_model = os.getenv("GROQ_MODEL") or os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
    default_idx = available_models.index(env_model) if env_model in available_models else 0
    model = st.selectbox(
        "Groq Chat Model",
        available_models,
        index=default_idx,
        help="openai/gpt-oss-20b is fast & efficient; 120b is higher capacity",
    )
    top_k = st.slider("Retrieved Chunks (top-k)", 3, 12, 6)
    show_rewrite = st.checkbox("Show rewritten query", value=True)
    show_usage = st.checkbox("Show token usage", value=True)

    st.divider()
    st.markdown("## 📚 Knowledge Base")

    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

    # Discover available collections from ChromaDB
    def _get_collections():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=chroma_dir)
            cols = client.list_collections()
            names = [c.name if hasattr(c, "name") else str(c) for c in cols]
            # Ensure known default names are listed
            for fallback in ["fastapi_demo", "python_docs"]:
                if fallback not in names:
                    names.append(fallback)
            return sorted(names)
        except Exception:
            return ["fastapi_demo", "python_docs"]

    avail_cols = _get_collections()
    # Preferred selection: session state override (e.g. after fresh ingest), then env, then first available
    preferred = st.session_state.get("selected_collection") or os.getenv("COLLECTION_NAME", "fastapi_demo" if "fastapi_demo" in avail_cols else "python_docs")
    if preferred not in avail_cols:
        avail_cols.append(preferred)
        avail_cols = sorted(avail_cols)
    default_idx = avail_cols.index(preferred) if preferred in avail_cols else 0

    collection = st.selectbox(
        "Active Collection",
        avail_cols,
        index=default_idx,
        help="Select which ChromaDB collection to query",
    )

    # Auto-load stats when collection is selected or changes
    if "current_col" not in st.session_state or st.session_state.current_col != collection:
        st.session_state.current_col = collection
        try:
            from vectorstore.chroma_store import ChromaVectorStore
            from vectorstore.embeddings import TrackedEmbeddings
            emb = TrackedEmbeddings(model="BAAI/bge-small-en-v1.5")
            s = ChromaVectorStore(persist_dir=chroma_dir, collection_name=collection, embeddings=emb)
            st.session_state.kb_stats = s.collection_stats()
        except Exception:
            st.session_state.kb_stats = None

    if st.button("🔍 Load / Refresh KB Stats"):
        try:
            from vectorstore.chroma_store import ChromaVectorStore
            from vectorstore.embeddings import TrackedEmbeddings
            emb = TrackedEmbeddings(model="BAAI/bge-small-en-v1.5")
            s = ChromaVectorStore(persist_dir=chroma_dir, collection_name=collection, embeddings=emb)
            st.session_state.kb_stats = s.collection_stats()
        except Exception as e:
            st.error(f"Could not load KB: {e}")

    if st.session_state.kb_stats:
        domains_list = st.session_state.kb_stats.get("domains", [])
        domain_line = f"<br>🌐 <b>Source:</b> {', '.join(domains_list)}" if domains_list else ""
        st.markdown(f"""
        <div class="metric-card">
          📦 <b>Collection:</b> {st.session_state.kb_stats['collection']}<br>
          📄 <b>Chunks:</b> {st.session_state.kb_stats['document_count']:,}{domain_line}
        </div>
        """, unsafe_allow_html=True)

    # ── Ingest a New Website Panel ─────────────────────────────────────────
    with st.expander("🌐 Ingest a New Website", expanded=False):
        st.markdown("<p style='font-size: 0.85em; color: #8fa3b7;'>Crawl and index any public documentation or site into a new ChromaDB collection.</p>", unsafe_allow_html=True)
        new_url = st.text_input("Website URL", placeholder="https://docs.example.com/", key="new_ingest_url")
        max_p = st.number_input("Max Pages (demo cap: 50)", min_value=1, max_value=50, value=20, step=5, key="new_ingest_max_pages")
        custom_col = st.text_input("Collection Name (optional)", placeholder="auto-derived if empty", key="new_ingest_col")

        if st.button("🚀 Crawl & Index", use_container_width=True, key="btn_crawl_index"):
            target_url = new_url.strip()
            if not target_url or not target_url.startswith(("http://", "https://")):
                st.error("Please enter a valid URL starting with http:// or https://")
            else:
                # Derive safe collection name
                if custom_col.strip():
                    coll_name = re.sub(r"[^a-zA-Z0-9_-]", "_", custom_col.strip())
                else:
                    p = urlparse(target_url)
                    host_slug = p.netloc.replace(".", "_")
                    sub_slug = p.path.strip("/").replace("/", "_")
                    raw_slug = f"{host_slug}_{sub_slug}" if sub_slug else host_slug
                    coll_name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_slug)[:35]

                with st.status(f"Ingesting {target_url}...", expanded=True) as status_box:
                    try:
                        from crawler.web_crawler import WebCrawler
                        from crawler.content_processor import ContentProcessor
                        from vectorstore.chroma_store import ChromaVectorStore
                        from vectorstore.embeddings import TrackedEmbeddings

                        # 1. Crawl
                        status_box.update(label=f"Crawling {target_url} (up to {int(max_p)} pages)...")
                        crawler = WebCrawler(start_url=target_url, max_pages=int(max_p), max_depth=3, crawl_delay=0.3)
                        pages = crawler.crawl(show_progress=False)

                        if not pages:
                            status_box.update(label="Crawl finished with 0 pages", state="error")
                            st.error(f"Could not fetch any pages from `{target_url}`. The site may block crawlers or the URL may be unreachable.")
                        else:
                            # 2. Extract & Chunk
                            status_box.update(label=f"Extracting content from {len(pages)} pages...")
                            processor = ContentProcessor(chunk_size=800, chunk_overlap=150)
                            docs = processor.process(pages)

                            if not docs:
                                status_box.update(label="0 text chunks extracted", state="error")
                                st.error("Pages were fetched, but no readable text content could be extracted.")
                            else:
                                # 3. Embed & Store
                                status_box.update(label=f"Embedding {len(docs)} chunks locally with BGE ($0)...")
                                emb = TrackedEmbeddings(model="BAAI/bge-small-en-v1.5")
                                store = ChromaVectorStore(persist_dir=chroma_dir, collection_name=coll_name, embeddings=emb)
                                store.add_documents(docs, batch_size=100)

                                status_box.update(label=f"✓ Indexed {len(pages)} pages ({len(docs)} chunks) into '{coll_name}'", state="complete")
                                st.success(f"Successfully indexed into `{coll_name}`!")

                                # Auto-select new collection and refresh UI
                                st.session_state["selected_collection"] = coll_name
                                st.session_state.current_col = coll_name
                                st.session_state.kb_stats = store.collection_stats()
                                st.rerun()

                    except Exception as exc:
                        status_box.update(label=f"Ingestion failed: {type(exc).__name__}", state="error")
                        st.error(f"Ingestion error: {exc}")

    st.divider()
    st.markdown("## 💰 Session Stats")

    if "tracker" in st.session_state and st.session_state.tracker:
        t = st.session_state.tracker
        st.markdown(f"""
        <div class="metric-card">
          🔢 <b>Queries:</b> {len(t.history)}<br>
          🪙 <b>LLM tokens:</b> {t.total_query_llm_tokens:,}<br>
          💵 <b>Session cost:</b> ${t.total_query_cost_usd:.4f}
        </div>
        """, unsafe_allow_html=True)

        if len(t.history) > 0:
            st.markdown("**Cost Projections:**")
            for n in [100, 1_000, 10_000]:
                proj = t.projection(n)
                cost = proj[f"estimated_cost_{n}_queries"]
                st.markdown(f"  - {n:,} queries → **${cost:.2f}**")
    else:
        st.markdown("*No queries yet*")


# ── Main area header ───────────────────────────────────────────────────────
def render_header(collection_name: str, kb_stats: dict | None):
    """Render generic title, subtitle, and dynamic source information in one place."""
    st.markdown("# 🌐 Website RAG Agent")
    count = kb_stats.get("document_count", 0) if kb_stats else 0
    domains = kb_stats.get("domains", []) if kb_stats else []

    if count > 0:
        domain_info = f" (source: <code>{', '.join(domains)}</code>)" if domains else ""
        st.markdown(
            f'<p class="subtitle">Currently querying: <b>{collection_name}</b> — <b>{count:,}</b> chunks indexed{domain_info}</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<p class="subtitle" style="color: #e09f3e;">⚠️ Currently querying: <b>{collection_name}</b> — 0 chunks indexed (no data loaded yet)</p>',
            unsafe_allow_html=True,
        )

render_header(collection, st.session_state.kb_stats)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "tracker" not in st.session_state:
    st.session_state.tracker = None


# ── Agent initialisation ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG agent (downloading model on first run)...")
def load_agent(groq_api_key: str, model: str, top_k: int, chroma_dir: str, collection: str):
    from vectorstore.chroma_store import ChromaVectorStore
    from vectorstore.embeddings import TrackedEmbeddings
    from agent.graph import RAGAgent
    from agent.token_tracker import TokenTracker

    emb = TrackedEmbeddings()  # local BGE model, no API key
    store = ChromaVectorStore(persist_dir=chroma_dir, collection_name=collection, embeddings=emb)
    tracker = TokenTracker(model=model)
    agent = RAGAgent(
        vector_store=store,
        model=model,
        top_k=top_k,
        tracker=tracker,
        groq_api_key=groq_api_key,
    )
    return agent, tracker


def ensure_agent():
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar. Get one free at https://console.groq.com/keys")
        return False
    try:
        agent, tracker = load_agent(groq_api_key, model, top_k, chroma_dir, collection)
        st.session_state.agent = agent
        st.session_state.tracker = tracker
        return True
    except Exception as e:
        st.error(f"Failed to load agent: {e}")
        return False


# ── Render chat history ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-msg">🤖 <b>Agent:</b><br><br>{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            pills = "".join(f'<a href="{s}" target="_blank" class="source-pill">🔗 {s.split("/")[-1] or s.split("/")[-2]}</a>' for s in msg["sources"])
            st.markdown(f"**Sources:** {pills}", unsafe_allow_html=True)
        if show_usage and msg.get("usage_table"):
            with st.expander("📊 Token Usage"):
                st.code(msg["usage_table"])


# ── Chat input ─────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about this site's content..."):
    if not ensure_agent():
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🔍 Searching docs and generating answer..."):
        result = st.session_state.agent.ask(question)

    # Build response markdown
    answer_md = result["answer"]

    # Append assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_md,
        "sources": result.get("sources", []),
        "usage_table": result["usage"].summary_table() if result.get("usage") else None,
        "rewritten_query": result.get("rewritten_query", ""),
    })

    # Show rewritten query toast if enabled
    if show_rewrite and result.get("rewritten_query") and result["rewritten_query"] != question:
        st.toast(f"🔍 Search query: {result['rewritten_query']}", icon="✨")

    st.rerun()


# ── Empty state ────────────────────────────────────────────────────────────
if not st.session_state.messages:
    chunk_count = st.session_state.kb_stats.get("document_count", 0) if st.session_state.kb_stats else 0
    domains = st.session_state.kb_stats.get("domains", []) if st.session_state.kb_stats else []

    if chunk_count == 0:
        st.markdown(f"""
        <div style="text-align:center; padding: 50px 20px; color: #8fa3b7; background: #131722; border-radius: 12px; border: 1px dashed #2a354d; margin-top: 20px;">
          <div style="font-size: 2.8em; margin-bottom: 10px;">📭</div>
          <h3 style="color: #d1dced; margin-bottom: 8px;">No Knowledge Base Loaded</h3>
          <p style="max-width: 550px; margin: 0 auto 12px auto; font-size: 0.95em;">
            Collection <code>{collection}</code> currently has 0 indexed chunks.
          </p>
          <p style="font-size: 0.9em; color: #6a82a0;">
            Use <b>"🌐 Ingest a New Website"</b> in the sidebar to crawl and index a site, or select an existing populated collection (e.g. <code>fastapi_demo</code>).
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        source_desc = f"<code>{domains[0]}</code>" if domains else f"collection <code>{collection}</code>"
        st.markdown(f"""
        <div style="text-align:center; padding: 40px 20px; color: #7a9abf;">
          <div style="font-size: 2.8em; margin-bottom: 8px;">🌐</div>
          <h3 style="color: #c4d9f2; margin-bottom: 6px;">Ready to answer questions about the indexed site</h3>
          <p style="margin-bottom: 4px;">Ask anything about the content that has been crawled and indexed.</p>
          <p style="font-size:0.85em; color: #5a7599;">All answers are grounded exclusively in {source_desc} ({chunk_count:,} chunks).</p>
        </div>
        """, unsafe_allow_html=True)

        # Context-aware example questions
        st.markdown("**💡 Try asking:**")
        if "fastapi" in collection.lower() or any("fastapi" in d.lower() for d in domains):
            examples = [
                "How do path parameters work in FastAPI?",
                "How do query parameters with default values work?",
                "How do I define a request body with Pydantic?",
                "Does FastAPI natively support gRPC streaming?",
            ]
        elif "python" in collection.lower() or any("python" in d.lower() for d in domains):
            examples = [
                "What is a list comprehension and how do I use it?",
                "How does Python's garbage collector work?",
                "What is the difference between __str__ and __repr__?",
                "How do I use context managers with the 'with' statement?",
            ]
        else:
            examples = [
                "What are the main topics covered in this documentation?",
                "How do I get started with the basics?",
                "What are common configuration options or patterns?",
                "Provide an example of using the primary API or features.",
            ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                if ensure_agent():
                    st.session_state.messages.append({"role": "user", "content": ex})
                    with st.spinner("Thinking..."):
                        result = st.session_state.agent.ask(ex)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "usage_table": result["usage"].summary_table() if result.get("usage") else None,
                    })
                    st.rerun()
