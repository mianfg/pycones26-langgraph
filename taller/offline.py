"""Modo sin conexión (plan B): modelos de juguete que no llaman a ninguna API.

Sirven para seguir el taller si falla la wifi o tu clave: los grafos, las
herramientas, la recuperación y la memoria funcionan de verdad; solo el «LLM»
es un imitador extractivo basado en reglas. No esperes respuestas brillantes.

Actívalo en `.env` con `LLM_MODEL=offline`.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from typing import Any, Literal, get_args, get_origin

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from taller.retrieval import tokenize

LEARN_HINTS = (
    "te aviso",
    "desde el",
    "a partir de",
    "recuerda",
    "apunta",
    "aprende",
    "desde ahora",
    "ten en cuenta",
    "ahora es",
    "ha cambiado",
)
PREF_HINTS = ("prefiero", "respóndeme", "respondeme", "llámame", "llamame", "mi equipo", "trabajo en")


def _text(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else message.text


def _best_sentences(question: str, context: str, n: int = 2) -> str:
    q = set(tokenize(question))
    tag, scored = "", []
    for line in context.splitlines():
        if line.startswith("[") and "]" in line:
            tag = line[: line.index("]") + 1]
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            overlap = len(q & set(tokenize(sentence)))
            if overlap:
                scored.append((overlap, sentence.strip(), tag))
    if not scored:
        return "No he encontrado información sobre eso en la documentación."
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:n]
    return " ".join(s for _, s, _ in top) + (f" {top[0][2]}" if top[0][2] else "")


def _default_for(annotation: Any, question: str) -> Any:
    origin = get_origin(annotation)
    if annotation is bool:
        return True
    if origin is Literal:
        return get_args(annotation)[0]
    if annotation in (int, float):
        return 0
    if origin is list:
        return []
    if origin is not None and type(None) in get_args(annotation):
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _fill(annotation, question)
    return question


def _fill(schema: type[BaseModel], question: str) -> BaseModel:
    values = {name: _default_for(field.annotation, question) for name, field in schema.model_fields.items()}
    return schema.model_validate(values)


class OfflineChatModel(BaseChatModel):
    """Imitador de LLM: decide herramientas por palabras clave y responde extrayendo frases."""

    tools: list[dict] = []

    @property
    def _llm_type(self) -> str:
        return "offline-extractivo"

    def bind_tools(self, tools: list, **kwargs: Any) -> OfflineChatModel:
        return OfflineChatModel(tools=[convert_to_openai_tool(t)["function"] for t in tools])

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any):
        def run(messages: Any) -> BaseModel:
            msgs = messages.to_messages() if hasattr(messages, "to_messages") else messages
            question = _text(msgs[-1]) if isinstance(msgs, list) else str(msgs)
            return _fill(schema, question)

        return RunnableLambda(run)

    def _pick_tool(self, text: str) -> dict | None:
        lowered = text.lower()
        names = {t["name"]: t for t in self.tools}
        for hints, keyword in ((PREF_HINTS, "preferencia"), (LEARN_HINTS, "aprend")):
            if any(h in lowered for h in hints):
                for name, tool in names.items():
                    if keyword in name:
                        return tool
        if "hora" in lowered and "hora_actual" in names:
            return names["hora_actual"]
        for name, tool in names.items():
            if "buscar" in name or "search" in name:
                return tool
        if len(self.tools) == 1:  # una sola herramienta: la usa (como haría un modelo pequeño)
            return self.tools[0]
        return None

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        last = messages[-1]
        text = _text(last)
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        question = _text(humans[-1]) if humans else text

        if match := re.search(r"responde solo con la palabra:\s*(\w+)", text, flags=re.IGNORECASE):
            reply = AIMessage(content=match.group(1))
        elif self.tools and isinstance(last, HumanMessage) and (tool := self._pick_tool(text)):
            params = list(tool.get("parameters", {}).get("properties", {}))
            call = {
                "name": tool["name"],
                "args": {params[0]: text} if params else {},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
            reply = AIMessage(content="", tool_calls=[call])
        elif isinstance(last, ToolMessage):
            context = "\n".join(_text(m) for m in messages if isinstance(m, ToolMessage))
            reply = AIMessage(content=f"(modo offline) {_best_sentences(question, context)}")
        else:
            context = "\n".join(_text(m) for m in messages[:-1]) + "\n" + text.replace(question, "")
            reply = AIMessage(content=f"(modo offline) {_best_sentences(question, context)}")
        return ChatResult(generations=[ChatGeneration(message=reply)])


class HashingEmbeddings(Embeddings):
    """«Embeddings» léxicos por hashing de palabras: deterministas, sin red y sin semántica."""

    def __init__(self, dims: int = 256):
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for tok in tokenize(text):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
