"""
Central configuration for the News Intelligence Assistant.

All tunable parameters live here and can be overridden via environment
variables (typically set in a local `.env` file). Nothing in this module
performs I/O beyond reading the environment, so it is safe to import from
anywhere in the project.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    """Read a boolean flag from the environment ('1', 'true', 'yes' -> True)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to `default` on bad input."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to `default` on bad input."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# API keys / model selection
# ---------------------------------------------------------------------------

# Kept for backward compatibility with any tooling/docs that still reference
# Claude. The active generation engine for this project is Gemini (below).
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "news_articles")

# Distance metric used by the Chroma collection. "cosine" gives a bounded,
# well-understood distance (0 = identical, 2 = opposite) which makes the
# relevance threshold below meaningful and easy to tune. This only takes
# effect the first time a collection is created; an already-existing
# collection keeps whatever metric it was created with.
DISTANCE_METRIC = os.getenv("DISTANCE_METRIC", "cosine")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = _get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 120)

# ---------------------------------------------------------------------------
# Retrieval / hybrid routing
# ---------------------------------------------------------------------------

# Default number of source articles to retrieve per question.
TOP_K_DEFAULT = _get_int("TOP_K_DEFAULT", 6)

# How many candidate chunks to pull from Chroma per requested top_k, before
# deduplication/merging collapses them down to distinct articles.
RETRIEVAL_FANOUT = _get_int("RETRIEVAL_FANOUT", 5)

# Maximum cosine distance for a retrieved chunk to be considered "relevant".
# Lower = stricter (only near-exact matches pass). With cosine distance,
# 0.0 means identical embeddings and ~1.0+ means largely unrelated content.
# Tune this empirically for your embedding model / corpus.
RAG_DISTANCE_THRESHOLD = _get_float("RAG_DISTANCE_THRESHOLD", 0.45)

# Minimum number of relevant chunks required before we trust the news index
# enough to answer from it rather than falling back to Gemini's own
# knowledge.
RAG_MIN_RELEVANT_CHUNKS = _get_int("RAG_MIN_RELEVANT_CHUNKS", 1)

# Toggle the hybrid fallback behaviour entirely. If False, the assistant
# behaves like the original news-only RAG bot.
ENABLE_GEMINI_FALLBACK = _get_bool("ENABLE_GEMINI_FALLBACK", True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Default RSS feeds (used when no --feeds file is supplied)
# ---------------------------------------------------------------------------

DEFAULT_RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://rss.cnn.com/rss/cnn_topstories.rss",
    "https://feeds.npr.org/1001/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://moxie.foxnews.com/google-publisher/latest.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://techcrunch.com/feed/",
    "https://www.espn.com/espn/rss/news",
]
