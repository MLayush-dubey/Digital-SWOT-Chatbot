import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import requests
import config

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital SWOT Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Helper: check Ollama connectivity ────────────────────────────────────────

def check_ollama_status() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        resp = requests.get(config.OLLAMA_BASE_URL, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ── Helper: load RAGChain (cached) ───────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_rag_chain():
    """Initialise and cache the RAGChain."""
    from chatbot.rag_chain import RAGChain
    return RAGChain()


# ── Session state initialisation ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role": ..., "content": ...}

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None  # quick-action prompt waiting to fire


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Digital SWOT Chatbot")
    st.markdown(
        "Ask me anything about **Digital SWOT's** services, team, or how we can "
        "help grow your business online."
    )
    st.divider()

    # Ollama status indicator
    ollama_ok = check_ollama_status()
    if ollama_ok:
        st.success("Ollama: Connected", icon="✅")
    else:
        st.error("Ollama: Not running", icon="❌")
        st.caption(
            "Start Ollama with `ollama serve` and ensure the required models are pulled."
        )

    st.divider()

    # Quick action buttons
    st.markdown("**Quick questions:**")

    quick_actions = [
        "What services do you offer?",
        "How can I contact you?",
        "Tell me about your AI solutions",
    ]

    for action in quick_actions:
        if st.button(action, use_container_width=True):
            st.session_state.pending_prompt = action

    st.divider()

    # Clear chat
    if st.button("Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.caption("Powered by Ollama + ChromaDB + LangChain")


# ── Main chat UI ──────────────────────────────────────────────────────────────

st.markdown("## Digital SWOT AI Assistant")
st.caption("Ask me about our services, team, or how to get in touch.")

# Render existing conversation
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "💬"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Show sources if present in the message metadata
        if msg.get("sources"):
            st.markdown(
                "<small>**Sources:** " +
                " | ".join(f"[{s}]({s})" for s in msg["sources"]) +
                "</small>",
                unsafe_allow_html=True,
            )


# ── Handle pending quick-action prompt ───────────────────────────────────────

user_input: str | None = None

if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ── Chat input ────────────────────────────────────────────────────────────────

typed_input = st.chat_input("Ask me anything about Digital SWOT...")
if typed_input:
    user_input = typed_input


# ── Process user input ────────────────────────────────────────────────────────

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="💬"):
        st.markdown(user_input)

    # Guard: Ollama must be running
    if not check_ollama_status():
        error_msg = (
            "Ollama is not running. Please start it with `ollama serve` and ensure "
            f"`{config.OLLAMA_LLM_MODEL}` and `{config.OLLAMA_EMBED_MODEL}` are pulled."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": error_msg, "sources": []}
        )
        with st.chat_message("assistant", avatar="🤖"):
            st.error(error_msg)
        st.stop()

    # Load RAG chain
    try:
        rag = load_rag_chain()
    except Exception as e:
        error_msg = (
            f"Failed to load the knowledge base: {e}\n\n"
            "Please run `python ingestion/ingest.py` to build the ChromaDB first."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": error_msg, "sources": []}
        )
        with st.chat_message("assistant", avatar="🤖"):
            st.error(error_msg)
        st.stop()

    # Build chat history (exclude the message we just appended)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    # Generate response with spinner
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result = rag.query(user_input, chat_history=history)
                answer = result["answer"]
                sources = result.get("sources", [])
            except Exception as e:
                answer = (
                    f"Sorry, something went wrong while generating a response. "
                    f"Please try again or contact us at info@digitalswot.ae.\n\n"
                    f"_(Error: {e})_"
                )
                sources = []

        st.markdown(answer)

        if sources:
            st.markdown(
                "<small>**Sources:** " +
                " | ".join(f"[{s}]({s})" for s in sources) +
                "</small>",
                unsafe_allow_html=True,
            )

    # Persist assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

    # Rerun to refresh chat (handles quick-action flow cleanly)
    st.rerun()
