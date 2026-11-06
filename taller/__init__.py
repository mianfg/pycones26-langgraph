"""Código base del taller «Introducción a LangGraph: construyendo agentes que aprenden» (PyConES 2026).

Cada módulo corresponde a una pieza que se construye en los notebooks. Los notebooks
importan desde aquí lo que se hizo en secciones anteriores, para que cada sección
pueda empezarse sin haber completado las previas.
"""

from taller.config import get_embeddings, get_llm

__all__ = ["get_embeddings", "get_llm"]
