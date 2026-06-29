"""
Local vector store for ingested news articles, backed by ChromaDB with a
SentenceTransformers embedding function.

Responsibilities:
    * Chunk article text for embedding.
    * Persist chunks (with metadata) into a local Chroma collection.
    * Retrieve the most relevant chunks for a question, deduplicated and
      merged back into per-article hits with a relevance distance attached
      so the caller (RAG engine) can decide whether the news index actually
      has something useful to say.
"""

import logging
import re
from typing import Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

import config

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
    """Split `text` into overlapping fixed-size chunks.

    Args:
        text: Raw text to split.
        chunk_size: Max characters per chunk. Defaults to config.CHUNK_SIZE.
        overlap: Characters of overlap between consecutive chunks. Defaults
            to config.CHUNK_OVERLAP.

    Returns:
        A list of text chunks (empty list if `text` is empty/whitespace).
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    text = (text or "").strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def _normalize_title(title: str) -> str:
    """Lowercase + strip punctuation/whitespace, for near-duplicate detection."""
    title = (title or "").lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    return re.sub(r"\s+", " ", title).strip()


class NewsVectorStore:
    """Thin wrapper around a persistent Chroma collection of news chunks."""

    def __init__(
        self,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Initialize (or open) the persistent Chroma collection.

        Args:
            path: Filesystem path for the persistent Chroma client.
            collection_name: Name of the Chroma collection to use.
            embedding_model: SentenceTransformers model name for embeddings.
        """
        self.client = chromadb.PersistentClient(path=path or config.CHROMA_PATH)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model or config.EMBEDDING_MODEL
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name or config.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": config.DISTANCE_METRIC},
        )

    def add_articles(self, articles: List[dict]) -> int:
        """Chunk, embed, and persist a list of articles.

        Articles already present (by chunk id) are skipped, so this is safe
        to call repeatedly on overlapping batches.

        Args:
            articles: List of article dicts with at least
                `id`, `title`, `url`, `source`, `published`, and
                `content`/`summary`.

        Returns:
            Number of newly added chunks.
        """
        ids: List[str] = []
        docs: List[str] = []
        metadatas: List[dict] = []

        for article in articles:
            text = article.get("content") or article.get("summary") or ""
            chunks = chunk_text(text)

            if not chunks:
                continue

            for i, chunk in enumerate(chunks):
                chunk_id = f"{article['id']}_{i}"

                ids.append(chunk_id)
                docs.append(chunk)
                metadatas.append(
                    {
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", ""),
                        "published": article.get("published", ""),
                        "article_id": article["id"],
                        "chunk_index": i,
                    }
                )

        if not ids:
            logger.info("No new chunks to add.")
            return 0

        existing_ids = set()
        try:
            existing_results = self.collection.get(ids=ids)
            existing_ids = set(existing_results.get("ids", []))
        except Exception:
            logger.debug("Could not pre-check existing ids; proceeding without skip-check.")

        new_ids, new_docs, new_meta = [], [], []
        for _id, _doc, _meta in zip(ids, docs, metadatas):
            if _id not in existing_ids:
                new_ids.append(_id)
                new_docs.append(_doc)
                new_meta.append(_meta)

        if not new_ids:
            logger.info("All chunks already present in the vector store.")
            return 0

        batch_size = 256
        for start in range(0, len(new_ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=new_ids[start:end],
                documents=new_docs[start:end],
                metadatas=new_meta[start:end],
            )

        logger.info("Added %d new chunks to the vector store.", len(new_ids))
        return len(new_ids)

    def query(self, question: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve, deduplicate, and merge the most relevant article hits.

        Multiple chunks from the same article are merged into a single hit
        (ordered by their original chunk index, not retrieval order), and
        near-duplicate articles (same normalized title) are collapsed to the
        single best-scoring instance. Each returned hit also carries a
        `distance` field (lower = more relevant) so the caller can apply a
        relevance threshold.

        Args:
            question: The user's natural-language question.
            top_k: Number of distinct articles to return. Defaults to
                config.TOP_K_DEFAULT.

        Returns:
            A list of hit dicts, sorted by ascending distance (best first),
            each with keys: text, title, url, source, published, distance,
            article_id.
        """
        top_k = top_k or config.TOP_K_DEFAULT

        if self.count() == 0:
            return []

        # Over-fetch raw chunks so that, after merging chunks from the same
        # article, we still end up with `top_k` distinct articles.
        n_results = max(top_k * config.RETRIEVAL_FANOUT, top_k)

        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results,
            )
        except Exception as e:
            logger.error("Chroma query failed: %s", e)
            return []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Group chunks by article_id (fallback to url, then title) so that
        # multiple chunks from one article merge into a single coherent hit.
        articles: Dict[str, dict] = {}

        for doc, meta, dist in zip(docs, metas, distances):
            key = meta.get("article_id") or meta.get("url") or meta.get("title", "unknown")

            if key not in articles:
                articles[key] = {
                    "chunks": [],
                    "title": meta.get("title", ""),
                    "url": meta.get("url", ""),
                    "source": meta.get("source", ""),
                    "published": meta.get("published", ""),
                    "article_id": meta.get("article_id", key),
                    "distance": dist,
                }

            articles[key]["chunks"].append((meta.get("chunk_index", 0), doc))
            articles[key]["distance"] = min(articles[key]["distance"], dist)

        # Merge each article's chunks in original document order, not
        # retrieval-score order, so the merged text reads coherently.
        hits = []
        for article in articles.values():
            ordered_chunks = sorted(article["chunks"], key=lambda c: c[0])
            merged_text = "\n\n".join(text for _, text in ordered_chunks)
            hits.append(
                {
                    "text": merged_text,
                    "title": article["title"],
                    "url": article["url"],
                    "source": article["source"],
                    "published": article["published"],
                    "distance": article["distance"],
                    "article_id": article["article_id"],
                }
            )

        # Collapse near-duplicate articles (same normalized title, e.g. the
        # same wire story syndicated by multiple sources), keeping the
        # lowest-distance instance of each.
        deduped: Dict[str, dict] = {}
        for hit in hits:
            norm_title = _normalize_title(hit["title"]) or hit["article_id"]
            if norm_title not in deduped or hit["distance"] < deduped[norm_title]["distance"]:
                deduped[norm_title] = hit

        final_hits = sorted(deduped.values(), key=lambda h: h["distance"])
        return final_hits[:top_k]

    def count(self) -> int:
        """Return the total number of chunks currently stored."""
        return self.collection.count()
