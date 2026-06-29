# 📰 News Intelligence Assistant

A RAG-based assistant that fetches the latest news and answers your questions
with cited sources. Runs locally — only your question and retrieved excerpts
are sent to Claude; embeddings and storage are fully local and free.

## How it works

```
RSS feeds / NewsAPI  -->  chunk + embed (local)  -->  ChromaDB (local vector store)
                                                              |
                                                              v
                          your question  -->  retrieve top-k chunks  -->  Claude  -->  answer + [1][2][3] sources
```

- **Ingestion** (`news_fetcher.py`): pulls live articles from RSS feeds (no key
  required) and optionally NewsAPI.org for broader coverage.
- **Vector store** (`vector_store.py`): chunks article text, embeds it locally
  with `sentence-transformers`, and persists it in a local ChromaDB instance.
- **RAG engine** (`rag_engine.py`): retrieves the most relevant chunks for a
  question and asks Claude to answer strictly from that context, citing
  sources as `[1]`, `[2]`, etc.

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY (required)
# NEWSAPI_KEY is optional — leave blank to use RSS feeds only
```

Get an Anthropic API key at https://console.anthropic.com/
Get a free NewsAPI key (optional) at https://newsapi.org/

## Usage

### 1. Ingest the latest news

```bash
python ingest.py
```

Options:
```bash
python ingest.py --no-fulltext            # faster: index RSS summaries only
python ingest.py --feeds feeds.txt         # use a custom feed list
python ingest.py --query "AI regulation"   # also pull NewsAPI articles on a topic
```

Run this periodically (e.g. via cron every hour) to keep the index fresh:
```bash
# crontab -e
0 * * * * cd /path/to/news_intel_assistant && /path/to/venv/bin/python ingest.py >> ingest.log 2>&1
```

### 2. Ask questions

```bash
python ask.py "What's the latest on the EU AI Act?"
```

Or run interactively:
```bash
python ask.py
Ask> What happened in the markets today?
```

### 3. Or use the web UI

```bash
streamlit run app.py
```

This gives you a browser UI to fetch news and ask questions, with sources
shown as clickable links.

## Customizing

- **Feeds**: edit `config.DEFAULT_RSS_FEEDS` or pass `--feeds yourfile.txt`.
- **Embedding model**: change `EMBEDDING_MODEL` in `.env` to any
  `sentence-transformers` model (e.g. `all-mpnet-base-v2` for higher quality,
  slower).
- **Claude model**: change `CLAUDE_MODEL` in `.env`.
- **Chunk size / retrieval depth**: tune `CHUNK_SIZE`, `CHUNK_OVERLAP`,
  `TOP_K_DEFAULT` in `config.py`.

## Notes & limitations

- Full-text extraction (`newspaper3k`) works on most news sites but can fail
  on paywalled or JS-heavy pages — it falls back to the RSS summary in that case.
- The assistant answers **only from indexed articles** — if nothing relevant
  is indexed yet, it will tell you rather than guessing.
- ChromaDB data persists in `./data/chroma` between runs; delete that folder
  to reset the index.
- This is a starting point, not a production deployment — for production use
  consider adding deduplication-by-content (not just URL), rate limiting,
  and a real task scheduler instead of cron.

## Project structure

```
news_intel_assistant/
├── config.py          # settings (feeds, model names, chunking params)
├── news_fetcher.py     # RSS + NewsAPI fetching and normalization
├── vector_store.py      # chunking, local embeddings, ChromaDB persistence
├── rag_engine.py        # retrieval + Claude answer generation with citations
├── ingest.py            # CLI: fetch news -> index it
├── ask.py               # CLI: ask questions, get sourced answers
├── app.py               # Streamlit web UI
├── feeds.txt            # example custom feed list
├── requirements.txt
├── .env.example
└── data/                # local ChromaDB storage (created on first run)
```
