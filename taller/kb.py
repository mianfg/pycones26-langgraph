"""Carga de la base de conocimiento: Markdown → fragmentos (chunks) por sección."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

from taller.config import DATA_DIR

KB_DIR = DATA_DIR / "conocimiento"


def split_markdown(text: str, source: str) -> list[Document]:
    """Divide un Markdown en un fragmento por cada sección `##`.

    Cada fragmento lleva el título del documento y de la sección: así el texto
    se entiende por sí solo y la recuperación (sobre todo BM25) funciona mejor.
    """
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else source
    chunks = []
    for section in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        heading, _, body = section.partition("\n")
        body = body.strip()
        if not body:
            continue
        chunks.append(
            Document(
                page_content=f"{doc_title} — {heading.strip()}\n\n{body}",
                metadata={"source": source, "section": heading.strip()},
                id=f"{source}#{len(chunks)}",
            )
        )
    return chunks


def load_knowledge_base(path: Path = KB_DIR) -> list[Document]:
    """Lee todos los `.md` de la carpeta y devuelve sus fragmentos."""
    docs = []
    for file in sorted(path.glob("*.md")):
        docs.extend(split_markdown(file.read_text(encoding="utf-8"), file.name))
    return docs


def format_docs(docs: list[Document]) -> str:
    """Formatea los documentos para el prompt, con una etiqueta citable."""
    return "\n\n".join(
        f"[{d.metadata.get('source', '?')} › {d.metadata.get('section', '')}]\n{d.page_content}" for d in docs
    )
