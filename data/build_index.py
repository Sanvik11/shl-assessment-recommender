"""
Optional convenience script: pre-builds and sanity-checks the Gemini+FAISS
retriever against data/catalog.json (the full scraped catalog) so you can
verify embeddings/index construction before deploying.

The FastAPI app builds this index in-process on startup (see app/retriever.py
get_retriever), so this script isn't required to run the service -- it's
useful for a quick standalone check after scraping:

    export GEMINI_API_KEY=...
    python data/build_index.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.catalog import Catalog
from app.retriever import GeminiFaissRetriever


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running this script.")

    catalog = Catalog.load()
    print(f"Loaded {len(catalog)} assessments. Embedding + indexing with Gemini...")
    retriever = GeminiFaissRetriever(catalog, api_key)
    print("Index built. Running a couple of sanity queries:\n")

    for query in [
        "mid-level Java developer who works with stakeholders",
        "personality assessment for a graduate hire",
        "SQL and data warehousing knowledge test",
    ]:
        results = retriever.search(query, k=5)
        print(f"Query: {query!r}")
        for a, score in results:
            print(f"  {score:.3f}  {a.name}  ({a.test_type})")
        print()


if __name__ == "__main__":
    main()
