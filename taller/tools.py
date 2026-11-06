"""Herramientas del agente.

Las de la sección 2 son funciones puras. Las de las secciones 4 y 5 necesitan
acceder al *store* de LangGraph y al contexto de la ejecución (quién habla):
lo reciben con `runtime: ToolRuntime`, un parámetro que LangGraph inyecta y
que el LLM no ve.
"""

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langgraph.types import interrupt

from taller.kb import format_docs
from taller.retrieval import HybridRetriever, reciprocal_rank_fusion

MADRID = ZoneInfo("Europe/Madrid")
KNOWLEDGE_NS = ("conocimiento",)


# --- Sección 2: herramientas sencillas ----------------------------------------


@tool
def hora_actual() -> str:
    """Devuelve la fecha, el día de la semana y la hora actuales en Madrid."""
    now = datetime.now(MADRID)
    dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    return f"{dias[now.weekday()]} {now:%Y-%m-%d %H:%M} (hora peninsular)"


@tool
def dias_laborables(inicio: str, fin: str) -> int:
    """Cuenta los días laborables (lunes a viernes) entre dos fechas, ambas incluidas.

    Args:
        inicio: fecha inicial en formato AAAA-MM-DD.
        fin: fecha final en formato AAAA-MM-DD.
    """
    start, end = date.fromisoformat(inicio), date.fromisoformat(fin)
    days = (end - start).days + 1
    return sum((start + timedelta(d)).weekday() < 5 for d in range(days))


# --- Secciones 4 y 5: herramientas con memoria --------------------------------


@dataclass
class Context:
    """Contexto de la ejecución: datos que no cambian durante una conversación."""

    user_id: str = "anonimo"


def learned_to_documents(items) -> list[Document]:
    """Convierte elementos del store en `Document` para fusionarlos con la base de conocimiento."""
    return [
        Document(
            page_content=item.value["texto"],
            metadata={"source": "aprendido", "section": f"{item.value['fecha']} por {item.value['autor']}"},
            id=f"aprendido#{item.key}",
        )
        for item in items
    ]


def make_memory_tools(retriever: HybridRetriever, require_approval: bool = False) -> dict:
    """Crea las herramientas del agente que aprende, en un diccionario por nombre.

    `require_approval=True` (sección 5) pide confirmación humana con `interrupt()`
    antes de guardar conocimiento nuevo: evita que cualquiera «envenene» la memoria.
    """

    @tool
    def buscar_documentacion(consulta: str, runtime: ToolRuntime[Context]) -> str:
        """Busca en la documentación interna de Cuac & Co. y en lo aprendido en conversaciones anteriores.

        Úsala para cualquier pregunta sobre la empresa: políticas, procesos, herramientas, errores...
        """
        docs = retriever.search(consulta, k=4)
        learned = learned_to_documents(runtime.store.search(KNOWLEDGE_NS, query=consulta, limit=3))
        fused = [d for d, _ in reciprocal_rank_fusion([docs, learned])][:5]
        return format_docs(fused) or "No hay resultados."

    @tool
    def aprender_hecho(hecho: str, runtime: ToolRuntime[Context]) -> str:
        """Guarda un hecho nuevo o actualizado sobre la empresa para futuras conversaciones.

        Úsala cuando el usuario te informe de algo que no está en la documentación
        o que la contradice (por ejemplo, un cambio de proceso). Escribe el hecho
        como una frase autocontenida, con fechas si las hay.
        """
        # interrupt() pausa el grafo y lo guarda en el checkpointer. Al reanudar, la herramienta
        # vuelve a ejecutarse desde el principio: por eso no hay efectos secundarios antes de esta línea.
        if require_approval:
            decision = interrupt(
                {
                    "accion": "aprender_hecho",
                    "hecho": hecho,
                    "pregunta": "¿Apruebas guardar este conocimiento?",
                },
                # Con response_schema (un objeto), Studio muestra un formulario con las opciones en vez
                # de un editor de texto. Por código se reanuda igual con Command(resume="si").
                response_schema={
                    "type": "object",
                    "properties": {"decision": {"type": "string", "enum": ["si", "no"]}},
                    "required": ["decision"],
                },
            )
            if isinstance(decision, dict):  # respuesta del formulario de Studio
                decision = decision.get("decision")
            if decision != "si":
                return "Una persona revisora ha rechazado guardar este hecho. No se ha guardado."
        today = datetime.now(MADRID).date().isoformat()
        autor = runtime.context.user_id if runtime.context else "anonimo"
        value = {"texto": hecho, "autor": autor, "fecha": today}
        runtime.store.put(KNOWLEDGE_NS, uuid.uuid4().hex, value)
        return f"Guardado: {hecho}"

    @tool
    def guardar_preferencia(preferencia: str, runtime: ToolRuntime[Context]) -> str:
        """Guarda una preferencia del usuario sobre cómo quiere que le respondas (tono, formato, su equipo...)."""
        namespace = ("usuarios", runtime.context.user_id if runtime.context else "anonimo")
        current = runtime.store.get(namespace, "preferencias")
        prefs = (current.value["lista"] if current else []) + [preferencia]
        runtime.store.put(namespace, "preferencias", {"lista": prefs}, index=False)
        return f"Preferencia guardada: {preferencia}"

    return {
        t.name: t
        for t in (buscar_documentacion, aprender_hecho, guardar_preferencia, hora_actual, dias_laborables)
    }
