"""Streamlit UI for the Hybrid News Intelligence Assistant."""

import streamlit as st

import config
from news_fetcher import fetch_latest_news
from vector_store import NewsVectorStore
from rag_engine import NewsRAGEngine, SOURCE_NEWS

st.set_page_config(page_title="News Intelligence Assistant", page_icon="📰", layout="wide")

st.title("📰 News Intelligence Assistant")
st.caption(
    "Hybrid RAG-powered Q&A — answers from live indexed news when relevant, "
    "and falls back to Gemini's general knowledge otherwise."
)


@st.cache_resource
def get_store() -> NewsVectorStore:
    return NewsVectorStore()


@st.cache_resource
def get_engine(_store: NewsVectorStore) -> NewsRAGEngine:
    return NewsRAGEngine(vector_store=_store)


store = get_store()

with st.sidebar:
    st.header("News Index")
    st.metric("Chunks indexed", store.count())

    st.divider()
    st.subheader("Refresh news")
    query = st.text_input("Optional topic filter (NewsAPI)", "")
    fulltext = st.checkbox("Extract full article text (slower, better quality)", value=True)

    if st.button("Fetch latest news", type="primary"):
        with st.spinner("Fetching and indexing latest news..."):
            articles = fetch_latest_news(query=query or None, fetch_full_text=fulltext)
            added = store.add_articles(articles)
        st.success(f"Fetched {len(articles)} articles, added {added} new chunks.")
        st.rerun()

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Sources to retrieve per question", 3, 12, config.TOP_K_DEFAULT)
    distance_threshold = st.slider(
        "Relevance threshold (lower = stricter)",
        min_value=0.05,
        max_value=1.5,
        value=config.RAG_DISTANCE_THRESHOLD,
        step=0.05,
        help=(
            "Maximum cosine distance for a retrieved article to count as "
            "'relevant'. If no retrieved article is this close to the "
            "question, the assistant falls back to Gemini's general "
            "knowledge instead of forcing an answer from unrelated news."
        ),
    )
    config.RAG_DISTANCE_THRESHOLD = distance_threshold

    use_fallback = st.checkbox(
        "Allow Gemini general-knowledge fallback",
        value=config.ENABLE_GEMINI_FALLBACK,
        help="If unchecked, the assistant behaves like a news-only bot again.",
    )
    config.ENABLE_GEMINI_FALLBACK = use_fallback

st.subheader("Ask a question")
question = st.text_input("e.g. What's the latest on interest rate decisions?", "")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving relevant articles and generating answer..."):
        engine = get_engine(store)
        result = engine.ask(question, top_k=top_k)

    if result["source"] == SOURCE_NEWS:
        st.success("**Answer Source:** 📰 News Database")
    else:
        st.info("**Answer Source:** 🤖 Gemini Knowledge")

    st.markdown("### Answer")
    st.write(result["answer"])

    if result["source"] == SOURCE_NEWS and result["sources"]:
        st.markdown("### Sources")
        for s in result["sources"]:
            published = f" · {s['published']}" if s["published"] else ""
            st.markdown(f"**[{s['n']}]** [{s['title']}]({s['url']}) — *{s['source']}*{published}")
