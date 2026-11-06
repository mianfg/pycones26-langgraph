"""Recuperación híbrida: BM25 (léxica) + embeddings (semántica) + fusión RRF.

Implementado a mano a propósito: son patrones reutilizables con cualquier librería
o base de datos vectorial (pgvector, Qdrant, Elasticsearch, OpenSearch...).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

STOPWORDS = set(
    """
    a al algo ante como con cual cuando de del desde donde el ella en entre es esa ese esta este hay
    la las le lo los mas me mi muy no o para pero por que qué se si sin sobre su sus tu un una uno
    unos y ya yo puedo puede hace hacer tengo tiene son ser estoy cuanto cuantos cuanta
    """.split()
)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    """Minúsculas, sin tildes, sin palabras vacías y con un «stemmer» de juguete.

    Mantiene códigos como `err-4012` como un único token: justo lo que la
    búsqueda semántica suele perder y BM25 encuentra sin problema.
    """
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", _strip_accents(text.lower()))
    stems = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        if len(tok) > 4 and not tok[-1].isdigit():
            tok = tok.rstrip("s") or tok
            tok = tok[:-1] if tok[-1] in "aeo" else tok
        stems.append(tok)
    return stems


class BM25Index:
    """BM25 clásico (Okapi) en ~30 líneas."""

    def __init__(self, docs: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[Document] = []
        self.add(docs)

    def add(self, docs: Sequence[Document]) -> None:
        self.docs.extend(docs)
        self._tf = [Counter(tokenize(d.page_content)) for d in self.docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(len(self._len), 1)
        df = Counter(term for tf in self._tf for term in tf)
        n = len(self.docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        terms = tokenize(query)
        scores = []
        for doc, tf, length in zip(self.docs, self._tf, self._len):
            score = 0.0
            for t in terms:
                if t in tf:
                    norm = 1 - self.b + self.b * length / self._avg
                    score += self._idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * norm)
            scores.append((doc, score))
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        return [(d, s) for d, s in ranked[:k] if s > 0]


class DenseIndex:
    """Búsqueda semántica con embeddings sobre un vector store en memoria."""

    def __init__(self, docs: Sequence[Document], embeddings: Embeddings):
        self.store = InMemoryVectorStore(embeddings)
        self.add(docs)

    def add(self, docs: Sequence[Document]) -> None:
        if docs:
            self.store.add_documents(list(docs))

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Document]], k: int = 60
) -> list[tuple[Document, float]]:
    """Fusiona varios rankings usando solo las posiciones (no las puntuaciones).

    score(doc) = Σ 1 / (k + posición). No hay que normalizar puntuaciones de
    sistemas distintos (BM25 y coseno no son comparables): por eso es tan robusto.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking, start=1):
            key = doc.id or doc.page_content
            by_id[key] = doc
            scores[key] = scores.get(key, 0.0) + 1 / (k + position)
    return [(by_id[key], s) for key, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class HybridRetriever:
    """BM25 + embeddings fusionados con RRF. Sin embeddings, degrada a solo BM25."""

    def __init__(self, docs: Sequence[Document], embeddings: Embeddings | None = None):
        self.bm25 = BM25Index(docs)
        self.dense = DenseIndex(docs, embeddings) if embeddings else None

    def add(self, docs: Sequence[Document]) -> None:
        self.bm25.add(docs)
        if self.dense:
            self.dense.add(docs)

    def search(self, query: str, k: int = 4, candidates: int = 10) -> list[Document]:
        rankings = [[d for d, _ in self.bm25.search(query, k=candidates)]]
        if self.dense:
            rankings.append([d for d, _ in self.dense.search(query, k=candidates)])
        return [doc for doc, _ in reciprocal_rank_fusion(rankings)[:k]]
