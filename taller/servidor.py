"""Cliente del Agent Server local (`uv run langgraph dev`), para usarlo desde los notebooks.

El servidor expone los grafos de `langgraph.json` por HTTP en http://127.0.0.1:2024 y
Studio se conecta a él. Desde Python se habla con él con `langgraph_sdk`: es la misma
API que tendría el grafo desplegado.
"""

from __future__ import annotations

import httpx
from langgraph_sdk import get_sync_client
from langgraph_sdk.client import SyncLangGraphClient

URL = "http://127.0.0.1:2024"
STUDIO = "https://eu.smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"


def cliente(url: str = URL) -> SyncLangGraphClient | None:
    """Devuelve el cliente si el servidor responde; si no, explica cómo arrancarlo y devuelve None."""
    try:
        httpx.get(f"{url}/ok", timeout=2).raise_for_status()
    except httpx.HTTPError:
        print(
            "El servidor local no responde. Arráncalo en otra terminal, desde la raíz del repositorio:\n"
            "    uv run studio\n"
            "(atajo de `langgraph dev --studio-url https://eu.smith.langchain.com`, ver taller/cli.py)\n"
            f"y abre Studio en {STUDIO}"
        )
        return None
    return get_sync_client(url=url)


def contenido(mensaje: dict) -> str:
    """Texto de un mensaje del servidor: `content` puede ser una cadena o una lista de bloques."""
    content = mensaje.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def preguntar(graph_id: str, texto: str, *, user_id: str | None = None, thread_id: str | None = None):
    """Envía un mensaje a un grafo del servidor y muestra herramientas y respuesta. Devuelve el estado final."""
    client = cliente()
    if client is None:
        return None
    if thread_id is None:
        thread_id = client.threads.create()["thread_id"]
    context = {"user_id": user_id} if user_id else None
    result = client.runs.wait(
        thread_id, graph_id, input={"messages": [{"role": "human", "content": texto}]}, context=context
    )
    for m in result["messages"]:
        if m["type"] == "ai" and m.get("tool_calls"):
            for call in m["tool_calls"]:
                print(f"  [herramienta] {call['name']}({call['args']})")
    if result.get("__interrupt__"):
        print(f"  [interrupción] {result['__interrupt__'][0]['value']}")
    else:
        print(contenido(result["messages"][-1]))
    print(f"  (hilo {thread_id[:8]}… · ábrelo en Studio para ver el recorrido)")
    return result
