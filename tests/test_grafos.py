import sqlite3

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command

from taller.agents import (
    build_basic_agent,
    build_learning_agent,
    build_memory_store,
    build_rag_graph,
    build_retriever,
)
from taller.tools import KNOWLEDGE_NS, Context


@pytest.fixture(scope="module")
def retriever():
    return build_retriever()


def test_agente_basico_usa_herramientas():
    out = build_basic_agent().invoke({"messages": [("human", "¿qué hora es?")]})
    tipos = [type(m).__name__ for m in out["messages"]]
    assert tipos == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]


def test_rag_responde_con_cita(retriever):
    out = build_rag_graph(retriever).invoke({"question": "¿Cómo hago rollback?"})
    assert out["attempts"] >= 1
    assert "despliegues.md" in out["answer"]


def test_el_agente_aprende_entre_conversaciones(retriever):
    store = build_memory_store()
    agente = build_learning_agent(retriever, checkpointer=InMemorySaver(), store=store)
    ana, luis = Context(user_id="ana"), Context(user_id="luis")
    agente.invoke(
        {"messages": [("human", "Te aviso: el canal de incidencias ahora es #incidencias-cuac")]},
        {"configurable": {"thread_id": "c1"}},
        context=ana,
    )
    aprendido = store.search(KNOWLEDGE_NS)
    assert len(aprendido) == 1 and aprendido[0].value["autor"] == "ana"

    out = agente.invoke(
        {"messages": [("human", "¿cuál es el canal de incidencias?")]},
        {"configurable": {"thread_id": "c2"}},
        context=luis,
    )
    assert "#incidencias-cuac" in out["messages"][-1].text


def test_aprobacion_humana_con_interrupt_y_sqlite(retriever, tmp_path):
    def abrir():
        checkpointer = SqliteSaver(sqlite3.connect(tmp_path / "cp.sqlite", check_same_thread=False))
        conn = sqlite3.connect(tmp_path / "store.sqlite", check_same_thread=False, isolation_level=None)
        store = build_memory_store(SqliteStore, conn=conn)
        store.setup()
        return build_learning_agent(
            retriever, checkpointer=checkpointer, store=store, require_approval=True
        ), store

    config = {"configurable": {"thread_id": "p1"}}
    agente, store = abrir()
    res = agente.invoke(
        {"messages": [("human", "Apunta esto: la demo general pasa a los viernes")]},
        config,
        context=Context("ana"),
        version="v2",
    )
    assert len(res.interrupts) == 1 and "viernes" in res.interrupts[0].value["hecho"]
    assert store.search(KNOWLEDGE_NS) == []

    agente, store = abrir()  # «reinicio»: nuevos objetos, mismos ficheros
    res = agente.invoke(Command(resume="no"), config, context=Context("ana"), version="v2")
    assert store.search(KNOWLEDGE_NS) == []
    assert "rechazado" in res.value["messages"][-2].text.lower()

    # Mismo hecho en otro hilo, esta vez aprobado: se guarda con autor y fecha.
    config2 = {"configurable": {"thread_id": "p2"}}
    agente.invoke(
        {"messages": [("human", "Apunta esto: la demo general pasa a los viernes")]},
        config2,
        context=Context("ana"),
        version="v2",
    )
    res = agente.invoke(Command(resume="si"), config2, context=Context("ana"), version="v2")
    guardados = store.search(KNOWLEDGE_NS)
    assert len(guardados) == 1 and "viernes" in guardados[0].value["texto"]
    assert guardados[0].value["autor"] == "ana" and guardados[0].value["fecha"]
    assert "guardado" in res.value["messages"][-2].text.lower()


def test_sin_store_falla_con_mensaje_claro(retriever):
    agente = build_learning_agent(retriever)
    with pytest.raises(RuntimeError, match="store"):
        agente.invoke({"messages": [("human", "hola")]}, context=Context("ana"))
