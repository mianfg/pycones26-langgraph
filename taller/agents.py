"""Soluciones de referencia de cada sección (los «puntos de recuperación»).

Si te quedas atrás, importa de aquí lo que necesites y sigue con la sección actual.
"""

from functools import cache
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from taller.config import get_embeddings, get_llm
from taller.kb import format_docs, load_knowledge_base
from taller.retrieval import HybridRetriever
from taller.tools import Context, dias_laborables, hora_actual, make_memory_tools

# --- Sección 2: primer agente -------------------------------------------------


def build_basic_agent(tools: list | None = None, checkpointer=None):
    """LLM + herramientas en bucle: el patrón ReAct construido a mano."""
    tools = tools or [hora_actual, dias_laborables]
    llm = get_llm().bind_tools(tools)

    def agent(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # → "tools" o END
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


# --- Sección 3: RAG integrado en el grafo -------------------------------------


@cache
def build_retriever() -> HybridRetriever:
    """Base de conocimiento de Cuac & Co. indexada con BM25 + embeddings.

    Con caché: un único índice por proceso. El Agent Server ejecuta `studio.py` una vez por grafo
    servido (tres veces); sin la caché, la base de conocimiento se vectorizaría tres veces al arrancar
    y, con la capa gratuita de Gemini (100 textos por minuto), el arranque fallaría con un 429.
    """
    return HybridRetriever(load_knowledge_base(), get_embeddings())


class RAGState(TypedDict):
    question: str
    query: str
    documents: list[Document]
    answer: str
    attempts: int


class Grade(BaseModel):
    """Evaluación de si los documentos bastan para responder."""

    suficiente: bool = Field(description="True si los documentos permiten responder a la pregunta")
    nueva_consulta: str = Field(
        description="Si no bastan, una consulta de búsqueda alternativa; si bastan, vacío"
    )


RAG_PROMPT = """Eres el asistente interno de Cuac & Co.. Responde en español usando SOLO el contexto.
Cita la fuente entre corchetes, por ejemplo [despliegues.md › Ventanas de despliegue].
Si el contexto no contiene la respuesta, dilo claramente.

Contexto:
{context}"""

GRADE_PROMPT = """Pregunta: {question}

Documentos recuperados:
{context}

¿Contienen estos documentos la información necesaria para responder? Si no, propón una consulta
de búsqueda distinta (sinónimos, términos más concretos)."""

MAX_ATTEMPTS = 2


def build_rag_graph(retriever: HybridRetriever | None = None):
    """RAG correctivo: recuperar → evaluar → (reformular y repetir) → responder."""
    retriever = retriever or build_retriever()
    llm = get_llm()
    grader = llm.with_structured_output(Grade)

    def retrieve(state: RAGState):
        query = state.get("query") or state["question"]
        return {"documents": retriever.search(query, k=4), "attempts": state.get("attempts", 0) + 1}

    def grade(state: RAGState) -> dict:
        verdict = grader.invoke(
            GRADE_PROMPT.format(question=state["question"], context=format_docs(state["documents"]))
        )
        # verdict es None si el modelo no emite la llamada estructurada: seguimos con lo que hay.
        return {"query": "" if verdict is None or verdict.suficiente else verdict.nueva_consulta}

    def route(state: RAGState) -> Literal["retrieve", "generate"]:
        if state["query"] and state["attempts"] < MAX_ATTEMPTS:
            return "retrieve"
        return "generate"

    def generate(state: RAGState):
        messages = [
            SystemMessage(RAG_PROMPT.format(context=format_docs(state["documents"]))),
            ("human", state["question"]),
        ]
        return {"answer": llm.invoke(messages).text}

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route)
    builder.add_edge("generate", END)
    return builder.compile()


# --- Secciones 4 y 5: agente que aprende --------------------------------------

SYSTEM_PROMPT = """Eres el asistente interno de Cuac & Co.. Hoy es {today}.

- Para cualquier pregunta sobre la empresa, usa `buscar_documentacion` antes de responder. No inventes.
- Cita las fuentes entre corchetes, tal y como aparecen en los resultados.
- Los resultados con fuente [aprendido] vienen de conversaciones anteriores. Si contradicen la
  documentación y son más recientes, prevalecen: indícalo en la respuesta.
- Si el usuario te informa de un dato nuevo o de un cambio sobre la empresa, guárdalo con `aprender_hecho`.
- Si el usuario expresa cómo prefiere que le respondas, guárdalo con `guardar_preferencia`.

Preferencias de este usuario:
{preferences}"""


def build_learning_agent(
    retriever: HybridRetriever | None = None,
    *,
    checkpointer=None,
    store=None,
    require_approval: bool = False,
):
    """Agente con RAG como herramienta y memoria a largo plazo en el *store*."""
    retriever = retriever or build_retriever()
    tools = list(make_memory_tools(retriever, require_approval=require_approval).values())
    llm = get_llm().bind_tools(tools)

    def agent(state: MessagesState, runtime: Runtime[Context]):
        if runtime.store is None:
            raise RuntimeError("Este agente necesita un store: compila con store=build_memory_store()")
        user_id = runtime.context.user_id if runtime.context else "anonimo"
        prefs = runtime.store.get(("usuarios", user_id), "preferencias")
        system = SYSTEM_PROMPT.format(
            today=hora_actual.invoke({}),
            preferences="\n".join(f"- {p}" for p in prefs.value["lista"]) if prefs else "(ninguna)",
        )
        return {"messages": [llm.invoke([SystemMessage(system), *state["messages"]])]}

    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer, store=store)


def build_memory_store(store_cls=None, **kwargs):
    """Store con búsqueda semántica si hay embeddings; si no, un store simple."""
    from langgraph.store.memory import InMemoryStore

    store_cls = store_cls or InMemoryStore
    embeddings = get_embeddings()
    if embeddings is None:
        return store_cls(**kwargs)
    dims = len(embeddings.embed_query("dimensiones"))
    return store_cls(index={"embed": embeddings, "dims": dims, "fields": ["texto"]}, **kwargs)
