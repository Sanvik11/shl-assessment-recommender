"""
Retrieval layer that grounds the agent in the actual catalog.

Two backends behind one interface (Retriever.search):

- GeminiFaissRetriever: embeds catalog + query text with Gemini's
  text-embedding-004 and does nearest-neighbour search with FAISS
  (IndexFlatIP over L2-normalized vectors == cosine similarity).
  Used in production (requires GEMINI_API_KEY + network).

- TfidfRetriever: scikit-learn TF-IDF + cosine similarity, pure local
  computation. Used when no API key / no network is available (dev,
  CI, and this sandbox), so the same agent code path can be exercised
  end-to-end without a live Gemini key.

Which one loads is decided once, in get_retriever(), based on env vars.
Everything downstream (app/agent.py) only depends on the search() method,
so swapping the backend is a one-line/env-var change, not a rewrite.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from app.catalog import Assessment, Catalog


class BaseRetriever:
    def search(self, query: str, k: int = 15) -> List[Tuple[Assessment, float]]:
        raise NotImplementedError


class TfidfRetriever(BaseRetriever):
    """Offline fallback: no external API calls, no network required."""

    def __init__(self, catalog: Catalog):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.catalog = catalog
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = [a.searchable_text() for a in catalog.assessments]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 15) -> List[Tuple[Assessment, float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        if not query.strip():
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        order = np.argsort(-sims)[:k]
        return [(self.catalog.assessments[i], float(sims[i])) for i in order if sims[i] > 0]


class GeminiFaissRetriever(BaseRetriever):
    """Production backend: Gemini embeddings + FAISS cosine search."""

    EMBED_MODEL = "models/text-embedding-004"

    def __init__(self, catalog: Catalog, api_key: str):
        import faiss
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.genai = genai
        self.catalog = catalog

        vectors = self._embed_batch(
            [a.searchable_text() for a in catalog.assessments], task_type="retrieval_document"
        )
        vectors = np.array(vectors, dtype="float32")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def _embed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        out = []
        for t in texts:
            resp = self.genai.embed_content(
                model=self.EMBED_MODEL, content=t, task_type=task_type
            )
            out.append(resp["embedding"])
        return out

    def search(self, query: str, k: int = 15) -> List[Tuple[Assessment, float]]:
        import faiss

        if not query.strip():
            return []
        q_vec = np.array(
            self._embed_batch([query], task_type="retrieval_query"), dtype="float32"
        )
        faiss.normalize_L2(q_vec)
        scores, idxs = self.index.search(q_vec, min(k, len(self.catalog)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.catalog.assessments[idx], float(score)))
        return results


_retriever_instance: BaseRetriever | None = None


def get_retriever(catalog: Catalog) -> BaseRetriever:
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    api_key = os.getenv("GEMINI_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "auto")

    use_gemini = provider == "gemini" or (provider == "auto" and bool(api_key))
    if use_gemini:
        try:
            _retriever_instance = GeminiFaissRetriever(catalog, api_key)
            return _retriever_instance
        except Exception as e:  # pragma: no cover - network/key issues
            print(f"[retriever] Falling back to TF-IDF retriever, Gemini init failed: {e}")

    _retriever_instance = TfidfRetriever(catalog)
    return _retriever_instance
