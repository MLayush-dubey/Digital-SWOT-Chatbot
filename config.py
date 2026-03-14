# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "gemma3:4b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ChromaDB
CHROMA_PERSIST_DIR = "./data/chroma_db"
COLLECTION_NAME = "digitalswot"

# Chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 4
