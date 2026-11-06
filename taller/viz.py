"""Visualización de grafos en el notebook."""

from IPython.display import Image, Markdown, display


def mostrar_grafo(graph, png: bool = False) -> None:
    """Dibuja un grafo compilado.

    Por defecto usa Mermaid en Markdown: JupyterLab lo dibuja sin conexión.
    Con `png=True` se genera una imagen a través de mermaid.ink (necesita internet);
    útil en editores que no dibujan Mermaid.
    """
    drawable = graph.get_graph()
    if png:
        display(Image(drawable.draw_mermaid_png()))
    else:
        display(Markdown(f"```mermaid\n{drawable.draw_mermaid()}\n```"))


def mostrar_mensajes(mensajes) -> None:
    """Imprime una conversación de forma legible: rol, texto y llamadas a herramientas.

    `pretty_print()` de LangChain enseña el contenido en bruto, que con algunos proveedores
    (Gemini 3, Anthropic) es una lista de bloques con firmas internas: ilegible en pantalla.
    """
    for m in mensajes:
        rol = {"human": "humano", "ai": "agente", "tool": "herramienta", "system": "sistema"}.get(
            m.type, m.type
        )
        if m.type == "tool":
            print(f"[{rol} {m.name}] {m.text[:300]}")
            continue
        for call in getattr(m, "tool_calls", None) or []:
            print(f"[{rol} → herramienta] {call['name']}({call['args']})")
        if m.text.strip():
            print(f"[{rol}] {m.text.strip()}")
