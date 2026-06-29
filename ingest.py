
import argparse
import os

from news_fetcher import fetch_latest_news
from vector_store import NewsVectorStore


def load_feed_list(path: str):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="Fetch and ingest latest news.")
    parser.add_argument("--feeds", type=str, default=None, help="Path to a text file of RSS feed URLs.")
    parser.add_argument("--query", type=str, default=None, help="Optional topic query for NewsAPI.")
    parser.add_argument("--no-fulltext", action="store_true", help="Skip full article extraction (faster).")
    args = parser.parse_args()

    feed_urls = None
    if args.feeds:
        if not os.path.exists(args.feeds):
            raise SystemExit(f"Feed list file not found: {args.feeds}")
        feed_urls = load_feed_list(args.feeds)

    articles = fetch_latest_news(
        query=args.query,
        feed_urls=feed_urls,
        fetch_full_text=not args.no_fulltext,
    )

    if not articles:
        print("No articles fetched. Check your network/feeds and try again.")
        return

    store = NewsVectorStore()
    store.add_articles(articles)
    print(f"Vector store now contains {store.count()} chunks total.")


if __name__ == "__main__":
    main()
