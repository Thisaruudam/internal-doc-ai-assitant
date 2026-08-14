"""Build the retrieval indexes from the corpus.

    uv run python scripts/index_corpus.py           # everything available
    uv run python scripts/index_corpus.py --bm25    # local fallback only

The BM25 fallback is built unconditionally because it needs no credentials, and
because a system that cannot reach Pinecone should still be able to answer.
Pinecone indexing is skipped with a clear message when no API key is configured,
rather than failing the whole seed — the difference between "degraded" and
"broken" is exactly what this system is meant to demonstrate.
"""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings
from app.observability.logging import configure_logging, get_logger
from app.retrieval.bm25_store import Bm25Index
from app.retrieval.chunking import chunk_corpus
from app.retrieval.corpus import CorpusError, load_documents

log = get_logger(__name__)

_PLACEHOLDER_KEYS = {"", "test-key-not-used", "your-key-here"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument("--bm25", action="store_true", help="build only the local BM25 index")
    parser.add_argument("--pinecone", action="store_true", help="build only the Pinecone indexes")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.observability)

    build_bm25 = args.bm25 or not args.pinecone
    build_pinecone = args.pinecone or not args.bm25

    try:
        documents = load_documents(settings.corpus_dir)
    except CorpusError as exc:
        print(f"\n  Corpus could not be loaded: {exc}", file=sys.stderr)
        print("  Run `make corpus` first.\n", file=sys.stderr)
        return 1

    chunks = chunk_corpus(documents)
    print(f"\n  Loaded {len(documents)} documents -> {len(chunks)} chunks")

    if build_bm25:
        index = Bm25Index.build(chunks)
        index.save(settings.bm25_index_dir)
        print(f"  BM25 fallback index written to {settings.bm25_index_dir}")

    if build_pinecone:
        api_key = settings.pinecone.api_key.get_secret_value()
        if api_key in _PLACEHOLDER_KEYS:
            print(
                "\n  Skipping Pinecone: PINECONE_API_KEY is not configured.\n"
                "  The BM25 fallback is built, so retrieval still works in\n"
                "  degraded mode. Set the key in .env and re-run to enable\n"
                "  hybrid search.\n"
            )
        else:
            import asyncio

            from app.retrieval.pinecone_store import PineconeStore

            store = PineconeStore(settings.pinecone, settings.gemini)
            print("  Provisioning Pinecone indexes (may take a minute on first run)...")
            store.ensure_indexes()
            upserted = asyncio.run(store.upsert_chunks(chunks))
            print(f"  Pinecone: upserted {upserted} chunks across both indexes")
            for name, stats in store.describe().items():
                print(f"    {name}: {stats.get('total_vector_count', '?')} vectors")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
