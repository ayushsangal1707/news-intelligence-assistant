import hashlib
from datetime import datetime, timezone

import feedparser
import requests

import config


def _make_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _parse_time(entry) -> str:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc).isoformat()
    return ""


def _extract_full_text(url: str) -> str:
    try:
        from newspaper import Article

        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception:
        return ""


def fetch_from_rss(feed_urls=None, max_per_feed: int = 15, fetch_full_text: bool = True):
    feed_urls = feed_urls or config.DEFAULT_RSS_FEEDS
    articles = []

    for feed_url in feed_urls:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"  [!] Failed to parse feed {feed_url}: {e}")
            continue

        source_name = parsed.feed.get("title", feed_url)

        for entry in parsed.entries[:max_per_feed]:
            url = entry.get("link", "")
            if not url:
                continue

            summary = entry.get("summary", "") or entry.get("description", "")
            content = ""
            if fetch_full_text:
                content = _extract_full_text(url)

            articles.append({
                "id": _make_id(url),
                "title": entry.get("title", "Untitled"),
                "url": url,
                "source": source_name,
                "published": _parse_time(entry),
                "summary": summary,
                "content": content or summary,
            })

    return articles


def fetch_from_newsapi(query: str = None, page_size: int = 30, fetch_full_text: bool = True):
    if not config.NEWSAPI_KEY:
        return []

    endpoint = "https://newsapi.org/v2/top-headlines" if not query else "https://newsapi.org/v2/everything"
    params = {
        "apiKey": config.NEWSAPI_KEY,
        "pageSize": page_size,
        "language": "en",
    }
    if query:
        params["q"] = query
    else:
        params["country"] = "us"

    try:
        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [!] NewsAPI request failed: {e}")
        return []

    articles = []
    for item in data.get("articles", []):
        url = item.get("url", "")
        if not url:
            continue
        content = ""
        if fetch_full_text:
            content = _extract_full_text(url)
        articles.append({
            "id": _make_id(url),
            "title": item.get("title", "Untitled"),
            "url": url,
            "source": (item.get("source") or {}).get("name", "NewsAPI"),
            "published": item.get("publishedAt", ""),
            "summary": item.get("description", "") or "",
            "content": content or item.get("content", "") or item.get("description", ""),
        })
    return articles


def fetch_latest_news(query: str = None, feed_urls=None, fetch_full_text: bool = True):
    print("Fetching from RSS feeds...")
    articles = fetch_from_rss(feed_urls=feed_urls, fetch_full_text=fetch_full_text)

    if config.NEWSAPI_KEY:
        print("Fetching from NewsAPI...")
        articles += fetch_from_newsapi(query=query, fetch_full_text=fetch_full_text)

    seen = set()
    deduped = []
    for a in articles:
        if a["id"] not in seen:
            seen.add(a["id"])
            deduped.append(a)

    print(f"Fetched {len(deduped)} unique articles.")
    return deduped


if __name__ == "__main__":
    items = fetch_latest_news(fetch_full_text=False)
    for a in items[:5]:
        print(f"- [{a['source']}] {a['title']} ({a['url']})")