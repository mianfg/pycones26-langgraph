"""Caché en disco de embeddings: cada texto se vectoriza una sola vez por máquina.

La capa gratuita de Gemini limita los embeddings a 100 textos por minuto y 1 000 al día.
Cada notebook vuelve a indexar los 34 fragmentos de la base de conocimiento al arrancar y
el servidor también; sin caché, tres o cuatro reinicios de kernel agotan la cuota diaria.
Con caché, la base de conocimiento cuesta 34 textos en todo el taller.

Se guarda en `.cache/embeddings.sqlite` (ignorado por git), con el nombre del modelo en la
clave: cambiar de proveedor no mezcla vectores de distintas dimensiones.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

from langchain_core.embeddings import Embeddings


class EmbeddingsCache(Embeddings):
    """Envuelve otro `Embeddings` y guarda cada vector en SQLite."""

    def __init__(self, inner: Embeddings, model: str, path: Path):
        self.inner, self.model = inner, model
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS vectores (clave TEXT PRIMARY KEY, vector TEXT NOT NULL)"
        )

    def _clave(self, tipo: str, texto: str) -> str:
        return f"{self.model}|{tipo}|{hashlib.sha256(texto.encode('utf-8')).hexdigest()}"

    def _leer(self, claves: list[str]) -> dict[str, list[float]]:
        if not claves:
            return {}
        marcas = ",".join("?" * len(claves))
        filas = self.conn.execute(f"SELECT clave, vector FROM vectores WHERE clave IN ({marcas})", claves)
        return {clave: json.loads(vector) for clave, vector in filas}

    def _guardar(self, pares: list[tuple[str, list[float]]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO vectores (clave, vector) VALUES (?, ?)",
            [(clave, json.dumps(vector)) for clave, vector in pares],
        )

    def _pendientes(self, tipo: str, textos: list[str]):
        claves = [self._clave(tipo, t) for t in textos]
        hechos = self._leer(list(set(claves)))
        unicos = list(dict.fromkeys(t for c, t in zip(claves, textos) if c not in hechos))
        return claves, hechos, unicos

    def _completar(self, tipo: str, hechos: dict, unicos: list[str], vectores: list[list[float]]) -> None:
        pares = [(self._clave(tipo, t), v) for t, v in zip(unicos, vectores)]
        self._guardar(pares)
        hechos.update(pares)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        claves, hechos, unicos = self._pendientes("doc", list(texts))
        if unicos:
            self._completar("doc", hechos, unicos, self.inner.embed_documents(unicos))
        return [hechos[c] for c in claves]

    def embed_query(self, text: str) -> list[float]:
        claves, hechos, unicos = self._pendientes("query", [text])
        if unicos:
            self._completar("query", hechos, unicos, [self.inner.embed_query(text)])
        return hechos[claves[0]]

    # Rutas async (las usa el Agent Server): el acceso a SQLite va a un hilo para no bloquear el bucle de eventos.
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        claves, hechos, unicos = await asyncio.to_thread(self._pendientes, "doc", list(texts))
        if unicos:
            vectores = await self.inner.aembed_documents(unicos)
            await asyncio.to_thread(self._completar, "doc", hechos, unicos, vectores)
        return [hechos[c] for c in claves]

    async def aembed_query(self, text: str) -> list[float]:
        claves, hechos, unicos = await asyncio.to_thread(self._pendientes, "query", [text])
        if unicos:
            vector = await self.inner.aembed_query(text)
            await asyncio.to_thread(self._completar, "query", hechos, unicos, [vector])
        return hechos[claves[0]]
