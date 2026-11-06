"""Prueba de extremo a extremo del Agent Server local (lo mismo que haremos en Studio, por HTTP).

    uv run langgraph dev            # en otra terminal
    uv run python scripts/probar_servidor.py [http://127.0.0.1:2024]

Recorre los tres grafos de langgraph.json con el SDK de LangGraph: el agente
básico con una herramienta, el RAG correctivo, y el agente que aprende con su
aprobación humana (interrupt → resume) y la memoria del store del servidor.
"""

import sys
import uuid

from langgraph_sdk import get_sync_client

from taller.servidor import contenido

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:2024"


def ultimo_texto(result: dict) -> str:
    return contenido(result["messages"][-1])


def main() -> int:
    client = get_sync_client(url=URL)
    graphs = sorted(
        {a["graph_id"] for a in client.assistants.search()}
    )  # puede haber asistentes creados en Studio
    print(f"Servidor {URL} · grafos: {', '.join(graphs)}")
    assert graphs == ["agente_basico", "agente_que_aprende", "rag"], graphs

    # 1. Agente básico: debe llamar a la herramienta hora_actual.
    thread = client.threads.create()
    result = client.runs.wait(
        thread["thread_id"],
        "agente_basico",
        input={"messages": [{"role": "human", "content": "¿Qué hora es?"}]},
    )
    tipos = [m["type"] for m in result["messages"]]
    print(f"\n[agente_basico] mensajes: {tipos}\n  → {ultimo_texto(result)[:120]}")
    assert "tool" in tipos, "el agente básico no ha usado ninguna herramienta"

    # 2. RAG correctivo: respuesta con cita a la fuente.
    result = client.runs.wait(None, "rag", input={"question": "¿Cómo hago rollback de un despliegue?"})
    print(
        f"\n[rag] intentos: {result['attempts']} · docs: {len(result['documents'])}\n  → {result['answer'][:160]}"
    )
    assert "despliegues.md" in result["answer"], "la respuesta del RAG no cita despliegues.md"

    # 3. Agente que aprende: aprender_hecho pide aprobación (interrupt) y luego guarda en el store.
    user = f"ana-{uuid.uuid4().hex[:6]}"
    thread = client.threads.create()
    hecho = "Te aviso: el canal de incidencias ahora es #incidencias-cuac"
    result = client.runs.wait(
        thread["thread_id"],
        "agente_que_aprende",
        input={"messages": [{"role": "human", "content": hecho}]},
        context={"user_id": user},
    )
    interrupts = result.get("__interrupt__", [])
    print(f"\n[agente_que_aprende] interrupciones pendientes: {len(interrupts)}")
    assert interrupts, "esperaba una interrupción de aprobación"
    print(f"  pregunta: {interrupts[0]['value']['pregunta']} · hecho: {interrupts[0]['value']['hecho'][:70]}")

    result = client.runs.wait(
        thread["thread_id"], "agente_que_aprende", command={"resume": "si"}, context={"user_id": user}
    )
    print(f"  → tras aprobar: {ultimo_texto(result)[:120]}")

    items = client.store.search_items(("conocimiento",), query="canal de incidencias", limit=3)["items"]
    print(f"  store (conocimiento): {len(items)} elemento(s) · último: {items[0]['value']['texto'][:70]!r}")
    assert any(
        "#incidencias-cuac" in it["value"]["texto"] for it in items
    ), "el hecho aprobado no está en el store"

    # 4. Conversación nueva, mismo usuario: la memoria a largo plazo sobrevive al hilo.
    thread = client.threads.create()
    result = client.runs.wait(
        thread["thread_id"],
        "agente_que_aprende",
        input={"messages": [{"role": "human", "content": "¿Cuál es el canal de incidencias?"}]},
        context={"user_id": user},
    )
    print(f"\n[memoria entre hilos] → {ultimo_texto(result)[:160]}")
    assert "#incidencias-cuac" in ultimo_texto(result), "el agente no ha recuperado lo aprendido"

    print("\nTodo correcto.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
