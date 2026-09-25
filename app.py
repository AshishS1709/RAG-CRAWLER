"""
app.py

Streamlit web UI for the Python Docs RAG Agent.

Launch with:
  streamlit run app.py
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Python Docs RAG Agent",
    page_icon="🐍",
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
    model = st.selectbox(
        "Groq Chat Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
        help="llama-3.1-8b-instant is fastest; 70b is smarter",
    )
    top_k = st.slider("Retrieved Chunks (top-k)", 3, 12, 6)
    show_rewrite = st.checkbox("Show rewritten query", value=True)
    show_usage = st.checkbox("Show token usage", value=True)

    st.divider()
    st.markdown("## 📚 Knowledge Base")

    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    collection = os.getenv("COLLECTION_NAME", "python_docs")

    if "kb_stats" not in st.session_state:
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
        st.markdown(f"""
        <div class="metric-card">
          📦 <b>Collection:</b> {st.session_state.kb_stats['collection']}<br>
          📄 <b>Chunks:</b> {st.session_state.kb_stats['document_count']:,}
        </div>
        """, unsafe_allow_html=True)

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


# ── Main area ──────────────────────────────────────────────────────────────
st.markdown("# 🐍 Python Docs RAG Agent")
st.markdown('<p class="subtitle">Grounded answers from docs.python.org/3/</p>', unsafe_allow_html=True)

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

    emb = TrackedEmbeddings()  # local Nomic model, no API key
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
        st.markdown(f'<div class="assistant-msg">🐍 <b>Agent:</b><br><br>{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            pills = "".join(f'<a href="{s}" target="_blank" class="source-pill">🔗 {s.split("/")[-1] or s.split("/")[-2]}</a>' for s in msg["sources"])
            st.markdown(f"**Sources:** {pills}", unsafe_allow_html=True)
        if show_usage and msg.get("usage_table"):
            with st.expander("📊 Token Usage"):
                st.code(msg["usage_table"])


# ── Chat input ─────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about Python..."):
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
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #4a6080;">
      <div style="font-size: 3em;">🐍</div>
      <h3 style="color: #6a8aaa;">Ready to answer your Python questions</h3>
      <p>Ask anything about Python's built-in functions, data structures, modules, syntax, and more.</p>
      <p style="font-size:0.85em;">All answers are grounded exclusively in the official Python 3 documentation.</p>
    </div>
    """, unsafe_allow_html=True)

    # Example questions
    st.markdown("**💡 Try asking:**")
    examples = [
        "What is a list comprehension and how do I use it?",
        "How does Python's garbage collector work?",
        "What is the difference between __str__ and __repr__?",
        "How do I use context managers with the 'with' statement?",
        "What are Python generators and when should I use them?",
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
