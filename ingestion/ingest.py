import os
import sys
import shutil

# Ensure project root is on the path so config.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw"
)

CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "chroma_db"
)


def infer_content_type(filename: str) -> str:
    """Infer content type from the filename (without extension)."""
    stem = os.path.splitext(filename)[0].lower()
    if stem == "core_info":
        return "core_info"
    if "about" in stem:
        return "about"
    if "contact" in stem:
        return "contact"
    if "case-study" in stem or "case_study" in stem:
        return "case_study"
    return "service"


def parse_metadata_from_file(filepath: str) -> dict:
    """Parse source_url, page_title, scrape_date from the file header."""
    metadata = {
        "source_url": "",
        "page_title": "",
        "scrape_date": "",
    }
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("source_url:"):
                    metadata["source_url"] = line.split("source_url:", 1)[1].strip()
                elif line.startswith("page_title:"):
                    metadata["page_title"] = line.split("page_title:", 1)[1].strip()
                elif line.startswith("scrape_date:"):
                    metadata["scrape_date"] = line.split("scrape_date:", 1)[1].strip()
                elif "=" * 10 in line:
                    # End of header block
                    break
    except Exception as e:
        print(f"  [WARN] Could not parse metadata from {filepath}: {e}")
    return metadata


def read_file_content(filepath: str) -> str:
    """Read the full text content of a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(raw_dir: str) -> list[dict]:
    """Load all .txt files from the raw directory."""
    documents = []
    if not os.path.exists(raw_dir):
        print(f"[ERROR] Raw data directory not found: {raw_dir}")
        print("Please run the scraper first: python scraper/scrape.py")
        sys.exit(1)

    txt_files = [f for f in os.listdir(raw_dir) if f.endswith(".txt") and f != ".gitkeep"]
    if not txt_files:
        print(f"[ERROR] No .txt files found in {raw_dir}")
        print("Please run the scraper first: python scraper/scrape.py")
        sys.exit(1)

    print(f"Found {len(txt_files)} .txt file(s) in {raw_dir}")

    for filename in sorted(txt_files):
        filepath = os.path.join(raw_dir, filename)
        content = read_file_content(filepath)
        file_metadata = parse_metadata_from_file(filepath)
        content_type = infer_content_type(filename)

        documents.append({
            "content": content,
            "metadata": {
                "source_url": file_metadata["source_url"],
                "page_title": file_metadata["page_title"],
                "scrape_date": file_metadata["scrape_date"],
                "content_type": content_type,
                "filename": filename,
            }
        })
        print(f"  Loaded: {filename} (type: {content_type})")

    return documents


def chunk_documents(documents: list[dict]) -> tuple[list[str], list[dict]]:
    """Split documents into chunks and return texts and metadatas."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    all_texts = []
    all_metadatas = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            meta = dict(doc["metadata"])
            meta["chunk_index"] = i
            all_metadatas.append(meta)

    return all_texts, all_metadatas


def build_vectorstore(texts: list[str], metadatas: list[dict], chroma_dir: str):
    """Delete existing ChromaDB if present, then create a new one."""
    if os.path.exists(chroma_dir) and os.path.isdir(chroma_dir):
        # Only delete if it contains actual chroma data (not just .gitkeep)
        contents = [f for f in os.listdir(chroma_dir) if f != ".gitkeep"]
        if contents:
            print(f"Deleting existing ChromaDB at: {chroma_dir}")
            shutil.rmtree(chroma_dir)
            os.makedirs(chroma_dir, exist_ok=True)

    print(f"Initialising OllamaEmbeddings (model: {config.OLLAMA_EMBED_MODEL}) ...")
    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )

    print(f"Embedding {len(texts)} chunks and storing in ChromaDB ...")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=config.COLLECTION_NAME,
        persist_directory=chroma_dir,
    )

    return vectorstore


def main():
    print("=" * 60)
    print("Digital SWOT — Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[1/3] Loading raw documents...")
    documents = load_documents(RAW_DIR)
    total_docs = len(documents)

    # Step 2: Chunk documents
    print(f"\n[2/3] Chunking {total_docs} document(s)...")
    texts, metadatas = chunk_documents(documents)
    total_chunks = len(texts)
    print(f"  Total chunks: {total_chunks}")

    # Step 3: Build vector store
    print(f"\n[3/3] Building ChromaDB at: {CHROMA_DIR}")
    vectorstore = build_vectorstore(texts, metadatas, CHROMA_DIR)

    # Summary
    collection_size = vectorstore._collection.count()
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"  Total documents loaded : {total_docs}")
    print(f"  Total chunks created   : {total_chunks}")
    print(f"  ChromaDB collection size: {collection_size}")
    print(f"  Collection name        : {config.COLLECTION_NAME}")
    print(f"  Persist directory      : {CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
