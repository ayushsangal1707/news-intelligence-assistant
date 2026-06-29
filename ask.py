
import argparse
import sys

from rag_engine import NewsRAGEngine, SOURCE_NEWS


def print_answer(result: dict):
    badge = "📰 News Database" if result.get("source") == SOURCE_NEWS else "🤖 Gemini Knowledge"

    print("\n" + "=" * 70)
    print(f"ANSWER  (Source: {badge})")
    print("=" * 70)
    print(result["answer"])

    if result["sources"]:
        print("\n" + "-" * 70)
        print("SOURCES")
        print("-" * 70)
        for s in result["sources"]:
            published = f" ({s['published']})" if s["published"] else ""
            print(f"[{s['n']}] {s['title']} — {s['source']}{published}")
            print(f"     {s['url']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Ask questions about ingested news.")
    parser.add_argument("question", nargs="?", default=None, help="Question to ask.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of source chunks to retrieve.")
    args = parser.parse_args()

    engine = NewsRAGEngine()

    if args.question:
        result = engine.ask(args.question, top_k=args.top_k)
        print_answer(result)
        return

    print("News Intelligence Assistant — interactive mode. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        result = engine.ask(question, top_k=args.top_k)
        print_answer(result)


if __name__ == "__main__":
    sys.exit(main())
