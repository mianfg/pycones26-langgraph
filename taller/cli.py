"""Comandos del proyecto (`[project.scripts]` en pyproject.toml).

    uv run studio

es un atajo de `uv run langgraph dev --studio-url https://eu.smith.langchain.com`: arranca el
Agent Server local y abre Studio en la región europea de LangSmith (donde están las cuentas del
taller). Sin `--studio-url`, `langgraph dev` abre la región de EE. UU. mientras las trazas estén
desactivadas. Cualquier argumento extra se pasa a `langgraph dev` (por ejemplo `--port 2025`).
"""

import subprocess
import sys

STUDIO_URL = "https://eu.smith.langchain.com"


def studio() -> None:
    comando = ["langgraph", "dev", "--studio-url", STUDIO_URL, *sys.argv[1:]]
    print("$", " ".join(comando))
    raise SystemExit(subprocess.call(comando))
