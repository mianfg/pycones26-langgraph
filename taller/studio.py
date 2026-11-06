"""Grafos del taller tal y como los sirve el Agent Server (`langgraph dev`).

`langgraph.json` apunta aquí. Es el mismo código de los notebooks con dos
diferencias que explican cómo funciona el servidor:

1. Los grafos se compilan **sin** checkpointer ni store: el servidor pone los
   suyos (en memoria con `langgraph dev`, Postgres en un despliegue) y los
   inyecta en `runtime.store` y en la persistencia de cada hilo.
2. La búsqueda semántica del store no se configura en Python sino en
   `langgraph.json` (`store.index`), que apunta a la función `embed` de abajo.

Trazas: el servidor las envía a LangSmith si en `.env` hay `LANGSMITH_TRACING=true`
(no hace falta código). Langfuse necesita un *callback* en cada invocación y el
servidor no lo admite al exportar el grafo: se usa desde los notebooks
(`taller.observabilidad.con_trazas`).

Arranca el servidor con `uv run studio` (atajo de `langgraph dev --studio-url
https://eu.smith.langchain.com`, ver cli.py) y abre Studio en
https://eu.smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
"""

from __future__ import annotations

from taller.agents import build_basic_agent, build_learning_agent, build_rag_graph, build_retriever
from taller.config import get_embeddings
from taller.tools import dias_laborables, hora_actual

# Dimensión fija del índice semántico del store del servidor (langgraph.json → store.index.dims).
# Los modelos de embeddings actuales (OpenAI text-embedding-3, Gemini embedding, Qwen3) admiten
# recortar el vector a menos dimensiones; los «embeddings» del modo offline tienen 256 y se rellenan.
STUDIO_DIMS = 768


async def embed(texts: list[str]) -> list[list[float]]:
    """Función de embeddings del store del servidor: debe ser async y aceptar una lista de textos."""
    embeddings = get_embeddings()
    if embeddings is None:
        # Proveedor sin embeddings (Anthropic): vector nulo, sin búsqueda semántica en el servidor.
        return [[0.0] * STUDIO_DIMS for _ in texts]
    vectors = await embeddings.aembed_documents(texts)
    return [_ajustar(v) for v in vectors]


def _ajustar(vector: list[float]) -> list[float]:
    vector = list(vector[:STUDIO_DIMS]) + [0.0] * max(0, STUDIO_DIMS - len(vector))
    norma = sum(x * x for x in vector) ** 0.5 or 1.0
    return [x / norma for x in vector]


# El servidor ejecuta este módulo una vez por grafo: build_retriever() lleva caché para no vectorizar tres veces.
_retriever = build_retriever()

# Bloque 3 · notebook 02: LLM + herramientas en un ciclo.
# Ejercicio 2: añade aquí tu herramienta nueva y el servidor la recarga solo.
TOOLS = [hora_actual, dias_laborables]
agente_basico = build_basic_agent(TOOLS)

# Bloque 4 · notebook 03: RAG correctivo (recuperar → evaluar → reintentar o responder).
rag = build_rag_graph(_retriever)

# Bloques 5 y 6 · notebooks 04 y 05: RAG como herramienta + memoria + aprobación humana.
agente_que_aprende = build_learning_agent(_retriever, require_approval=True)
