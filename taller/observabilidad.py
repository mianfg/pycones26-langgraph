"""Observabilidad opcional: LangSmith y Langfuse.

Ninguna de las dos es obligatoria para el taller. Si en `.env` hay claves, las
trazas de cada ejecución (prompts, llamadas a herramientas, tokens, latencia)
se envían a la plataforma correspondiente:

- **LangSmith** (de LangChain): basta con `LANGSMITH_TRACING=true` y
  `LANGSMITH_API_KEY`. No hay que tocar el código: LangChain lo detecta solo.
- **Langfuse** (código abierto, se puede autoalojar): hay que pasar un
  *callback handler* en cada llamada, `graph.invoke(..., config=con_trazas())`.
  Por eso funciona desde los notebooks y la demo, pero no en los grafos que sirve
  `langgraph dev` (el servidor exporta el grafo compilado, sin callbacks propios):
  ahí las trazas van a LangSmith.
"""

from __future__ import annotations

import os

from langchain_core.runnables import RunnableConfig

LANGFUSE_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def langsmith_activo() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def langfuse_activo() -> bool:
    return all(os.getenv(v) for v in LANGFUSE_VARS)


def callbacks() -> list:
    """Callbacks de trazado configurados en `.env` (hoy: solo Langfuse necesita uno)."""
    if not langfuse_activo():
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def con_trazas(config: RunnableConfig | None = None) -> RunnableConfig:
    """Devuelve `config` con los callbacks de trazado añadidos (si hay alguno)."""
    config = dict(config or {})
    config["callbacks"] = [*config.get("callbacks", []), *callbacks()]
    return config


def estado() -> str:
    """Resumen de qué plataformas de trazas están activas, para imprimir en los notebooks."""
    lineas = []
    if langsmith_activo():
        lineas.append(f"LangSmith: activo (proyecto {os.getenv('LANGSMITH_PROJECT', 'default')!r})")
    else:
        lineas.append(
            "LangSmith: inactivo (LANGSMITH_TRACING=true y LANGSMITH_API_KEY en .env para activarlo)"
        )
    if langfuse_activo():
        lineas.append(f"Langfuse: activo ({os.getenv('LANGFUSE_BASE_URL', 'https://cloud.langfuse.com')})")
    else:
        lineas.append("Langfuse: inactivo (LANGFUSE_PUBLIC_KEY y LANGFUSE_SECRET_KEY en .env para activarlo)")
    return "\n".join(lineas)
