# Digital SWOT — RAG Chatbot Implementation Plan (Phase 1)

## Overview

Build a domain-specific RAG chatbot for **Digital SWOT** (digitalswot.ae), a digital marketing agency in Dubai, UAE. The chatbot should answer questions about the company's services, contact info, case studies, and general business queries — grounded in the actual website content.

**This is Phase 1 — keep it lean.** We're building a working prototype, not a production system. We'll iterate from here.

---

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| LLM | **Ollama** (local) | Use `llama3.1:8b` or `mistral` — whatever runs well on dev machine |
| Embeddings | **Ollama embeddings** | Use `nomic-embed-text` via Ollama (keeps everything local) |
| Vector DB | **ChromaDB** | Lightweight, file-based, no server needed, pip install |
| RAG framework | **LangChain** | For chunking, retrieval, prompt assembly |
| Web scraping | **BeautifulSoup4 + requests** | Scrape digitalswot.ae pages |
| Backend | **Python** | Single script/module structure |
| Demo UI | **Streamlit** | Temporary frontend to test the chatbot |

---

## Project Structure

```
digitalswot-chatbot/
├── scraper/
│   └── scrape.py            # Scrapes website content
├── ingestion/
│   └── ingest.py            # Chunks, embeds, stores in ChromaDB
├── chatbot/
│   ├── rag_chain.py         # RAG pipeline (retrieve + generate)
│   └── prompts.py           # System prompt and prompt templates
├── app.py                   # Streamlit demo UI
├── data/
│   ├── raw/                 # Raw scraped HTML/text
│   └── chroma_db/           # ChromaDB persistent storage
├── config.py                # All config in one place
├── requirements.txt
└── README.md
```

---

## Step-by-Step Implementation

### Step 1: Config (`config.py`)

Centralize all settings:

```python
# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3.1:8b"          # or "mistral"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ChromaDB
CHROMA_PERSIST_DIR = "./data/chroma_db"
COLLECTION_NAME = "digitalswot"

# Chunking
CHUNK_SIZE = 600        # tokens roughly
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 4               # number of chunks to retrieve
```

### Step 2: Web Scraper (`scraper/scrape.py`)

Scrape these specific pages from digitalswot.ae:

```
PAGES_TO_SCRAPE = [
    "https://digitalswot.ae/",
    "https://digitalswot.ae/about-us/",
    "https://digitalswot.ae/contact-us/",
    "https://digitalswot.ae/services/",
    "https://digitalswot.ae/services/seo-services/",
    "https://digitalswot.ae/services/social-media-marketing/",
    "https://digitalswot.ae/services/paid-media-marketing/",
    "https://digitalswot.ae/services/growth-marketing/",
    "https://digitalswot.ae/services/content-and-production/",
    "https://digitalswot.ae/services/design-and-animation/",
    "https://digitalswot.ae/services/influencer-marketing/",
    "https://digitalswot.ae/services/affiliate-marketing/",
    "https://digitalswot.ae/services/web-design-and-development/",
    "https://digitalswot.ae/services/programmatic-advertising-with-dv360/",
    "https://digitalswot.ae/services/data-and-crm-management/",
    "https://digitalswot.ae/services/ai-solutions/",
    "https://digitalswot.ae/case-study/",
]
```

**Scraping logic:**
- Use `requests` + `BeautifulSoup4`
- Extract only the main content area (strip nav, footer, sidebar, scripts, styles)
- Save each page as a `.txt` file in `data/raw/` with the page slug as filename
- Also store metadata: `{"source_url": "...", "page_title": "...", "scrape_date": "..."}`
- Add a hardcoded block of essential business info that might not be clearly extractable from HTML:

```
HARDCODED_KNOWLEDGE = """
Company: Digital SWOT Marketing L.L.C.
Location: 1008, Grosvenor Business Tower, Tecom, Dubai, UAE
Phone: +971 522737711, 04 558 6320
Email: info@digitalswot.ae
WhatsApp: https://wa.me/971522737711
Working Hours: Monday to Friday, 9:00 AM to 6:00 PM (GST)
Website: https://digitalswot.ae
Social Media: Facebook, Instagram, TikTok, LinkedIn, YouTube, X (Twitter)

Digital SWOT is a full-service digital marketing agency based in Dubai with 10+ years of experience.
They have served 6800+ clients and completed 200+ successful digital projects.
They offer free initial consultations.

Services offered:
1. SEO Services
2. Social Media Marketing
3. Paid Media Marketing (Google Ads, Meta Ads, PPC)
4. Growth Marketing
5. Content & Production
6. Design & Animation
7. Influencer Marketing
8. Affiliate Marketing
9. Web Design & Development
10. Programmatic Advertising with DV360
11. Data & CRM Management
12. AI Solutions (chatbots, NLP, predictive analytics, ML, automation)

Notable clients include: Emaar Real Estate, Binance, Colgate, Valorgi, Immersive, and others.
Case studies available for: Valorgi, Immersive, Emaar Real Estate, Easy Map, Binance, Fooskha, Beirut Street, Colgate, Bougee Cafe, Ophir Properties.
"""
```

Save this hardcoded block as an additional document in `data/raw/core_info.txt`.

### Step 3: Ingestion Pipeline (`ingestion/ingest.py`)

This script reads all files from `data/raw/`, chunks them, embeds them, and stores in ChromaDB.

**Logic:**
1. Load all `.txt` files from `data/raw/`
2. Use LangChain's `RecursiveCharacterTextSplitter` with `chunk_size=600`, `chunk_overlap=100`
3. Tag each chunk with metadata: `source_url`, `content_type` (service/about/contact/case_study/core_info)
4. Use Ollama embeddings (`nomic-embed-text`) via LangChain's `OllamaEmbeddings`
5. Store in ChromaDB with persistence to `data/chroma_db/`
6. Print summary: total chunks, total documents, collection size

**Should be re-runnable:** If `data/chroma_db/` exists, delete and recreate (for Phase 1, full re-index is fine).

### Step 4: RAG Chain (`chatbot/rag_chain.py`)

The core retrieval-augmented generation pipeline.

**Logic:**
1. Load ChromaDB collection
2. On user query:
   - Embed the query using same Ollama embedding model
   - Retrieve top-k (4) most similar chunks from ChromaDB
   - Assemble the prompt using the template from `prompts.py`
   - Call Ollama LLM with the assembled prompt
   - Return the response + source URLs (from chunk metadata)
3. Maintain conversation history (last 5 exchanges) for multi-turn context
4. Use LangChain's `ChatOllama` for the LLM call

**Return format:**
```python
{
    "answer": "...",
    "sources": ["https://digitalswot.ae/services/seo-services/", ...],
    "num_chunks_retrieved": 4
}
```

### Step 5: System Prompt (`chatbot/prompts.py`)

```
SYSTEM_PROMPT = """You are the Digital SWOT AI Assistant — a helpful and professional chatbot for Digital SWOT, a leading digital marketing agency based in Dubai, UAE.

Your role:
- Answer questions about Digital SWOT's services, team, process, and expertise
- Help potential clients understand which services fit their needs
- Guide users toward booking a free consultation or contacting the team
- Be warm, professional, and concise

Rules:
- ONLY answer based on the context provided below. Do not make up information.
- If the context doesn't contain the answer, say: "I don't have specific information about that, but I'd recommend reaching out to our team directly at +971 522737711 or info@digitalswot.ae for more details."
- Never invent pricing — always direct to consultation for pricing questions
- Keep responses concise (2-4 sentences for simple questions, more for detailed service explanations)
- When mentioning services, briefly explain what Digital SWOT offers in that area
- End responses with a relevant call-to-action when appropriate (book consultation, visit service page, contact via WhatsApp)

Context from Digital SWOT's website:
{context}

Conversation history:
{chat_history}

User question: {question}

Answer:"""
```

### Step 6: Streamlit App (`app.py`)

A simple chat interface to test the bot.

**Features:**
- Chat-style UI using `st.chat_message`
- Persistent conversation history in `st.session_state`
- Show source URLs below each bot response (small text, clickable links)
- Sidebar with:
  - Title: "Digital SWOT Chatbot"
  - Short description
  - A "Clear Chat" button
  - Status indicator showing if Ollama is connected
  - Quick action buttons: "What services do you offer?", "How can I contact you?", "Tell me about your AI solutions"
- Handle errors gracefully (Ollama not running, ChromaDB empty, etc.)
- Show a spinner while the LLM generates

**Styling:**
- Use Digital SWOT's brand feel — nothing heavy, just set the page title and maybe a small logo/text header
- Bot avatar: 🤖, User avatar: 💬

### Step 7: Requirements (`requirements.txt`)

```
langchain
langchain-community
langchain-ollama
chromadb
beautifulsoup4
requests
streamlit
```

---

## How to Run (README.md content)

```bash
# 1. Install Ollama and pull models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Scrape the website
python -m scraper.scrape

# 4. Ingest into vector DB
python -m ingestion.ingest

# 5. Run the Streamlit app
streamlit run app.py
```

---

## Key Decisions & Rationale

- **Ollama over API:** Zero cost, full local control, good enough for a business chatbot. Can swap to Claude/GPT API later if needed.
- **ChromaDB over Pinecone/Qdrant:** No server to manage, file-based persistence, perfect for a prototype. Easy to migrate later.
- **Hardcoded core info:** Website HTML can be messy — critical business details (phone, email, hours, address) should be guaranteed in the knowledge base regardless of scraping quality.
- **Streamlit for now:** Fastest path to a testable UI. WordPress widget integration is Phase 2.
- **No Redis/caching yet:** Not needed at this scale. Add in Phase 2 if response times matter.

---

## Phase 2 (Future — not now)

- WordPress widget integration (JS bundle + FastAPI backend)
- Streaming responses
- Lead capture (collect name/email before or during chat)
- Arabic language support
- Analytics dashboard (popular questions, satisfaction ratings)
- Automated re-scraping on a schedule
- Response caching with Redis
- Better chunking strategies (semantic chunking)
- Re-ranking with a cross-encoder

---

## Testing Checklist (Phase 1)

After building, test these queries to verify the chatbot works:

1. "What services does Digital SWOT offer?" → Should list all 12 services
2. "How can I contact you?" → Should give phone, email, WhatsApp, address, hours
3. "Tell me about your SEO services" → Should describe their SEO offering
4. "How much do your services cost?" → Should NOT make up pricing, should redirect to consultation
5. "What is the weather today?" → Should politely decline (off-topic)
6. "Who are some of your clients?" → Should mention Emaar, Binance, Colgate, etc.
7. "Do you offer AI solutions?" → Should describe their AI/ML services
8. "Where is your office?" → Should give Grosvenor Business Tower, Tecom, Dubai
9. "What makes you different from other agencies?" → Should mention 10+ years, 6800+ clients, data-driven approach
10. "I want to grow my social media presence" → Should recommend their social media marketing service and suggest consultation
