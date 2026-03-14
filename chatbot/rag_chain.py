import os
import sys

# Ensure project root is on the path so config.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from chatbot.prompts import SYSTEM_PROMPT
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage

CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "chroma_db"
)


class RAGChain:
    """Retrieval-Augmented Generation pipeline for Digital SWOT chatbot."""

    def __init__(self):
        """Load the ChromaDB collection and initialise the LLM."""
        self.embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

        self.vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )

        self.llm = ChatOllama(
            model=config.OLLAMA_LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.3,
        )

        self.top_k = config.TOP_K

    def _format_chat_history(self, chat_history: list[dict]) -> str:
        """Format the conversation history as a readable string."""
        if not chat_history:
            return "No previous conversation."

        formatted = []
        for turn in chat_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                formatted.append(f"User: {content}")
            else:
                formatted.append(f"Assistant: {content}")

        return "\n".join(formatted)

    def _retrieve_context(self, question: str) -> tuple[str, list[str]]:
        """Retrieve relevant chunks from ChromaDB and return context + sources."""
        try:
            results = self.vectorstore.similarity_search(question, k=self.top_k)
        except Exception as e:
            return f"(Context retrieval failed: {e})", []

        if not results:
            return "No relevant context found.", []

        context_parts = []
        source_urls = []

        for doc in results:
            context_parts.append(doc.page_content)
            url = doc.metadata.get("source_url", "")
            if url and url not in source_urls:
                source_urls.append(url)

        context = "\n\n---\n\n".join(context_parts)
        return context, source_urls

    def query(self, question: str, chat_history: list[dict] = None) -> dict:
        """
        Run the RAG pipeline for a given question.

        Args:
            question: The user's question string.
            chat_history: List of dicts with keys 'role' ('user'/'assistant') and 'content'.

        Returns:
            dict with keys:
                - "answer": str
                - "sources": list of unique source URLs
                - "num_chunks_retrieved": int
        """
        if chat_history is None:
            chat_history = []

        # Step 1: Retrieve relevant context
        context, source_urls = self._retrieve_context(question)
        num_chunks = len(source_urls)

        # Step 2: Format chat history
        history_str = self._format_chat_history(chat_history)

        # Step 3: Assemble prompt
        prompt_text = SYSTEM_PROMPT.format(
            context=context,
            chat_history=history_str,
            question=question,
        )

        # Step 4: Call the LLM
        try:
            messages = [HumanMessage(content=prompt_text)]
            response = self.llm.invoke(messages)
            answer = response.content.strip()
        except Exception as e:
            answer = (
                f"I'm sorry, I encountered an error while generating a response. "
                f"Please try again or contact our team directly at info@digitalswot.ae. "
                f"(Error: {e})"
            )

        return {
            "answer": answer,
            "sources": source_urls,
            "num_chunks_retrieved": self.top_k,
        }
