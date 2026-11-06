from langchain_core.documents import Document

from taller.kb import load_knowledge_base, split_markdown
from taller.retrieval import BM25Index, HybridRetriever, reciprocal_rank_fusion, tokenize


def test_tokenize_conserva_codigos_y_quita_tildes():
    assert "err-4012" in tokenize("¿Qué significa el ERR-4012?")
    assert "despliegu" in tokenize("Despliegues") and "despliegu" in tokenize("despliegue")


def test_split_markdown_un_fragmento_por_seccion():
    docs = split_markdown("# Doc\n\n## A\n\ntexto a\n\n## B\n\ntexto b\n", "doc.md")
    assert [d.metadata["section"] for d in docs] == ["A", "B"]
    assert docs[0].page_content.startswith("Doc — A")
    assert docs[0].id == "doc.md#0"


def test_base_de_conocimiento_carga():
    docs = load_knowledge_base()
    assert len(docs) > 20
    assert all(d.metadata["source"].endswith(".md") for d in docs)


def test_bm25_encuentra_codigos_exactos():
    docs = load_knowledge_base()
    top, _ = BM25Index(docs).search("ERR-4012", k=1)[0]
    assert top.metadata["source"] == "errores_conocidos.md"


def test_rrf_premia_lo_que_aparece_en_ambos_rankings():
    a, b, c = (Document(page_content=x, id=x) for x in "abc")
    fused = reciprocal_rank_fusion([[a, c], [b, a]])
    assert [d.id for d, _ in fused] == ["a", "b", "c"]


def test_retriever_sin_embeddings_funciona_y_se_puede_ampliar():
    docs = load_knowledge_base()
    retriever = HybridRetriever(docs, embeddings=None)
    assert retriever.dense is None
    assert retriever.search("rollback", k=1)[0].metadata["section"].startswith("Marcha atrás")
    retriever.add([Document(page_content="Nuevo — Zanfona\n\nLa zanfona se afina los lunes", id="x#0")])
    assert retriever.search("zanfona", k=1)[0].id == "x#0"
