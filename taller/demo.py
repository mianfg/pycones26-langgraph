"""Demo de terminal del agente final: RAG híbrido + memoria persistente + aprobación humana.

    uv run python -m taller.demo --usuario ana

Comandos: /nuevo (conversación nueva), /memoria (ver lo aprendido), /salir.
"""

import argparse
import sqlite3
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command

from taller.agents import build_learning_agent, build_memory_store, build_retriever
from taller.config import ROOT, ruta_memoria
from taller.tools import KNOWLEDGE_NS, Context

DATA = ROOT / ".data"


def build_agent():
    DATA.mkdir(exist_ok=True)
    checkpointer = SqliteSaver(sqlite3.connect(DATA / "demo.sqlite", check_same_thread=False))
    store = build_memory_store(
        SqliteStore,
        conn=sqlite3.connect(ruta_memoria("demo_memoria"), check_same_thread=False, isolation_level=None),
    )
    store.setup()
    agent = build_learning_agent(
        build_retriever(), checkpointer=checkpointer, store=store, require_approval=True
    )
    return agent, store


def run_turn(agent, payload, config, ctx) -> None:
    """Ejecuta un turno mostrando herramientas y tokens en directo; gestiona las aprobaciones."""
    for modo, datos in agent.stream(payload, config, context=ctx, stream_mode=["messages", "updates"]):
        if modo == "messages":
            chunk, meta = datos
            if meta.get("langgraph_node") == "agent" and chunk.text:
                print(chunk.text, end="", flush=True)
        elif modo == "updates" and "agent" in datos:
            for call in datos["agent"]["messages"][-1].tool_calls:
                print(f"\n  [herramienta] {call['name']}({', '.join(map(str, call['args'].values()))})")
    print()

    for pending in agent.get_state(config).interrupts:
        answer = input(f"\n  [aprobación] ¿Guardar «{pending.value['hecho']}»? (s/n) ").strip().lower()
        run_turn(agent, Command(resume="si" if answer.startswith("s") else "no"), config, ctx)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--usuario", default="ana")
    args = parser.parse_args()

    agent, store = build_agent()
    ctx = Context(user_id=args.usuario)
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    print(f"Asistente de Cuac & Co. · usuario: {args.usuario} · /nuevo /memoria /salir\n")

    while True:
        try:
            text = input(f"{args.usuario}> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/salir":
            break
        if text == "/nuevo":
            config = {"configurable": {"thread_id": uuid.uuid4().hex}}
            print("  (conversación nueva: sin historial, pero con la memoria a largo plazo)\n")
            continue
        if text == "/memoria":
            for item in store.search(KNOWLEDGE_NS, limit=50):
                print(f"  · {item.value['texto']}  ({item.value['fecha']}, {item.value['autor']})")
            print()
            continue
        print("agente> ", end="")
        run_turn(agent, {"messages": [("human", text)]}, config, ctx)
        print()


if __name__ == "__main__":
    main()
