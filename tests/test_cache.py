"""La caché de embeddings evita repetir llamadas al proveedor."""

from langchain_core.embeddings import Embeddings

from taller.cache import EmbeddingsCache


class Contador(Embeddings):
    def __init__(self):
        self.llamadas = []

    def embed_documents(self, texts):
        self.llamadas.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        self.llamadas.append([text])
        return [float(len(text)), 2.0]


def test_cada_texto_se_vectoriza_una_vez(tmp_path):
    inner = Contador()
    cache = EmbeddingsCache(inner, "prueba", tmp_path / "emb.sqlite")
    assert cache.embed_documents(["a", "bb", "a"]) == [[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]]
    assert inner.llamadas == [["a", "bb"]]  # el duplicado no se pide dos veces
    assert cache.embed_documents(["bb", "ccc"]) == [[2.0, 1.0], [3.0, 1.0]]
    assert inner.llamadas[-1] == ["ccc"]  # solo el texto nuevo

    # Otro proceso (otro objeto) sobre el mismo fichero reutiliza lo guardado.
    inner2 = Contador()
    cache2 = EmbeddingsCache(inner2, "prueba", tmp_path / "emb.sqlite")
    assert cache2.embed_documents(["a", "bb", "ccc"]) == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert inner2.llamadas == []


def test_ruta_async_del_servidor_usa_la_misma_cache(tmp_path):
    import asyncio

    inner = Contador()
    cache = EmbeddingsCache(inner, "prueba", tmp_path / "emb.sqlite")
    cache.embed_documents(["a"])
    assert asyncio.run(cache.aembed_documents(["a", "bb"])) == [[1.0, 1.0], [2.0, 1.0]]
    assert inner.llamadas == [["a"], ["bb"]]


def test_consultas_y_modelos_no_se_mezclan(tmp_path):
    inner = Contador()
    cache = EmbeddingsCache(inner, "modelo-a", tmp_path / "emb.sqlite")
    assert cache.embed_query("a") == [1.0, 2.0]  # distinto del vector de documento
    assert cache.embed_query("a") == [1.0, 2.0] and len(inner.llamadas) == 1
    otro = EmbeddingsCache(Contador(), "modelo-b", tmp_path / "emb.sqlite")
    assert otro.embed_documents(["a"]) == [[1.0, 1.0]] and otro.inner.llamadas == [["a"]]
