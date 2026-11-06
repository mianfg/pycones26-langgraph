"""Modelos locales con Ollama: salida estructurada con respaldo por instrucciones.

En Apple Silicon, las etiquetas por defecto de Ollama (`gemma4:e4b`, `qwen3.5:4b`) descargan la
versión MLX del modelo, y el runner MLX de Ollama 0.40 no admite el parámetro `format`:
`with_structured_output` falla con «structured output is unavailable (status code: 501)».
Las etiquetas GGUF (`gemma4:e4b-it-q4_K_M`, `qwen3.5:4b-q4_K_M`) van por llama.cpp y sí lo admiten;
son las que recomienda el README. Para que el evaluador del RAG (bloque 4) funcione también con una MLX,
`ChatOllamaRobusto` intenta primero la salida estructurada nativa y, si el servidor la rechaza,
pide el JSON por instrucciones y lo valida con el mismo esquema Pydantic.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, convert_to_messages
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel


def _extraer_json(texto: str) -> str:
    """Se queda con el primer objeto JSON del texto (los modelos a veces lo envuelven en ```json)."""
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    return match.group(0) if match else texto


class ChatOllamaRobusto(ChatOllama):
    """`ChatOllama` cuyo `with_structured_output` funciona aunque el modelo no admita `format`."""

    def with_structured_output(self, schema: Any, **kwargs: Any):
        nativo = super().with_structured_output(schema, **kwargs)
        # El respaldo solo sabe devolver un objeto Pydantic: con otros esquemas o include_raw, nativo tal cual.
        if kwargs.get("include_raw") or not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return nativo
        parser = PydanticOutputParser(pydantic_object=schema)

        def con_instrucciones(entrada: Any) -> list:
            mensajes = (
                entrada.to_messages() if isinstance(entrada, PromptValue) else convert_to_messages(entrada)
            )
            return list(mensajes) + [HumanMessage(parser.get_format_instructions())]

        por_instrucciones = (
            RunnableLambda(con_instrucciones)
            | self
            | StrOutputParser()
            | RunnableLambda(_extraer_json)
            | parser
        )
        return nativo.with_fallbacks([por_instrucciones])
