"""
Hybrid RAG engine for the News Intelligence Assistant.

Routing logic:
    1. Retrieve candidate chunks from the local Chroma news index.
    2. Decide whether retrieval quality is good enough to trust, based on:
         - whether any chunks were retrieved at all,
         - their relevance distance vs. config.RAG_DISTANCE_THRESHOLD,
         - and the count of chunks passing that threshold vs.
           config.RAG_MIN_RELEVANT_CHUNKS.
    3a. If retrieval is good: answer strictly from the retrieved news
        excerpts, with inline [n] citations -> "news_database" source.
    3b. If retrieval is poor/empty (and fallback is enabled): answer from
        Gemini's own general knowledge, with no fabricated sources ->
        "gemini_knowledge" source.

The caller always gets a dict with `answer`, `sources`, and `source`, so the
UI can render the right citations and the right "Answer Source" badge.
"""

import logging
from typing import Dict, List, Optional

import google.generativeai as genai

import config
from vector_store import NewsVectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))


SOURCE_NEWS = "news_database"
SOURCE_GEMINI = "gemini_knowledge"


NEWS_SYSTEM_PROMPT = """You are a News Intelligence Assistant. Answer the user's \
question using ONLY the numbered news excerpts provided in the context below.

Rules:
- Base your answer strictly on the provided excerpts. Do not use outside knowledge.
- Cite sources inline using bracketed numbers like [1], [2] that map to the \
numbered sources listed in the context. Cite every factual claim you make.
- If different sources disagree or give conflicting information, point that \
out explicitly rather than silently picking one.
- Be concise, neutral, and factual. Do not speculate beyond what the \
excerpts support.
- Do not repeat the full source list at the end of your answer — it is \
displayed separately to the user.
"""

GENERAL_SYSTEM_PROMPT = """You are a knowledgeable, helpful general-purpose \
assistant. The user's question could not be answered from the local news \
index (no sufficiently relevant articles were found), so answer it using \
your own general knowledge instead.

Rules:
- Answer normally and helpfully, as you would for any general question.
- Do NOT invent or imply specific news article citations, source names, or \
publication dates — you have no retrieved sources for this answer.
- If the question is about very recent or fast-changing events you may not \
have reliable knowledge of, say so plainly rather than guessing.
- Be concise, neutral, and factual.
"""


def _build_context(hits: List[dict]) -> str:
    """Render retrieved hits into a numbered context block for the prompt."""
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(
            f"[{i}] Source: {hit['source']} | Title: {hit['title']} | "
            f"Published: {hit['published'] or 'unknown'}\n{hit['text']}\n"
        )
    return "\n".join(lines)


def _to_sources(hits: List[dict]) -> List[dict]:
    """Build the citation list shown in the UI, numbered to match the prompt."""
    return [
        {
            "n": i + 1,
            "title": hit["title"],
            "url": hit["url"],
            "source": hit["source"],
            "published": hit["published"],
        }
        for i, hit in enumerate(hits)
    ]


class NewsRAGEngine:
    """Hybrid question-answering engine: news RAG first, Gemini knowledge as fallback."""

    def __init__(self, vector_store: Optional[NewsVectorStore] = None) -> None:
        """Configure the Gemini client and attach (or create) a vector store.

        Raises:
            RuntimeError: if GEMINI_API_KEY is not configured.
        """
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )

        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        self.store = vector_store or NewsVectorStore()

    def _select_relevant_hits(self, hits: List[dict]) -> List[dict]:
        """Filter retrieved hits down to those passing the distance threshold."""
        return [h for h in hits if h["distance"] <= config.RAG_DISTANCE_THRESHOLD]

    def _should_use_news(self, hits: List[dict], relevant_hits: List[dict]) -> bool:
        """Decide whether retrieval quality is good enough to answer from news.

        Uses three signals together: did we retrieve anything at all, how
        many chunks pass the distance threshold, and (implicitly, via that
        threshold) how similar the best match actually is.
        """
        if not hits:
            return False
        return len(relevant_hits) >= config.RAG_MIN_RELEVANT_CHUNKS

    def _answer_from_news(self, question: str, hits: List[dict]) -> str:
        """Generate a grounded answer from retrieved news excerpts."""
        context = _build_context(hits)
        prompt = (
            f"{NEWS_SYSTEM_PROMPT}\n\n"
            f"Context (numbered news excerpts):\n\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer the question using ONLY the context above. "
            "Use inline citations like [1], [2]."
        )
        response = self.model.generate_content(prompt)
        return response.text

    def _answer_from_general_knowledge(self, question: str) -> str:
        """Generate an answer from Gemini's own knowledge (no news context)."""
        prompt = f"{GENERAL_SYSTEM_PROMPT}\n\nQuestion:\n{question}"
        response = self.model.generate_content(prompt)
        return response.text

    def ask(self, question: str, top_k: Optional[int] = None) -> Dict:
        """Answer a question, routing between the news index and Gemini.

        Args:
            question: The user's natural-language question.
            top_k: Number of distinct articles to retrieve. Defaults to
                config.TOP_K_DEFAULT.

        Returns:
            A dict with:
                answer (str): the generated answer text.
                sources (list[dict]): citation list, empty unless the
                    answer came from the news database.
                source (str): "news_database" or "gemini_knowledge".
                used_news (bool): convenience flag mirroring `source`.
        """
        question = (question or "").strip()
        if not question:
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "source": SOURCE_GEMINI,
                "used_news": False,
            }

        try:
            hits = self.store.query(question, top_k=top_k)
        except Exception as e:
            logger.error("Retrieval failed, falling back to general knowledge: %s", e)
            hits = []

        relevant_hits = self._select_relevant_hits(hits)
        use_news = self._should_use_news(hits, relevant_hits)

        if use_news:
            logger.info(
                "Routing to NEWS_DATABASE (%d/%d chunks passed threshold %.2f).",
                len(relevant_hits), len(hits), config.RAG_DISTANCE_THRESHOLD,
            )
            try:
                answer_text = self._answer_from_news(question, relevant_hits)
            except Exception as e:
                logger.error("Gemini news-grounded generation failed: %s", e)
                return {
                    "answer": f"Error generating response from the news index: {e}",
                    "sources": [],
                    "source": SOURCE_NEWS,
                    "used_news": True,
                }

            return {
                "answer": answer_text,
                "sources": _to_sources(relevant_hits),
                "source": SOURCE_NEWS,
                "used_news": True,
            }

        # --- Fallback path: no good news match ---
        if not config.ENABLE_GEMINI_FALLBACK:
            return {
                "answer": (
                    "The provided news excerpts do not contain enough "
                    "information to answer this question, and general-"
                    "knowledge fallback is disabled."
                ),
                "sources": [],
                "source": SOURCE_NEWS,
                "used_news": False,
            }

        logger.info(
            "Routing to GEMINI_KNOWLEDGE (0/%d chunks passed threshold %.2f).",
            len(hits), config.RAG_DISTANCE_THRESHOLD,
        )
        try:
            answer_text = self._answer_from_general_knowledge(question)
        except Exception as e:
            logger.error("Gemini general-knowledge generation failed: %s", e)
            answer_text = f"Error generating response: {e}"

        return {
            "answer": answer_text,
            "sources": [],
            "source": SOURCE_GEMINI,
            "used_news": False,
        }
