"""Configuración de modelos: un único punto para elegir proveedor.

Todo el taller usa `get_llm()` y `get_embeddings()`. El proveedor se elige en `.env`
con cadenas del tipo "proveedor:modelo", las mismas que entienden
`init_chat_model` e `init_embeddings` de LangChain.
"""

from __future__ import annotations

import os
import platform
import re
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import Embeddings, init_embeddings
from langchain_core.language_models import BaseChatModel

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datos"

load_dotenv(ROOT / ".env")

# Modelo por defecto de cada proveedor si en .env solo se indica el proveedor.
DEFAULT_LLM = {
    "openai": "openai:gpt-6-luna",
    "anthropic": "anthropic:claude-haiku-5-5",
    # gemini-3.5-flash-lite y no 3.8-flash: en la capa gratuita, 3.8/3.7/3.6-flash tienen 20 peticiones al día
    # (medido el 4-nov-2026); flash-lite tiene cientos y responde en segundos. Ver README.
    "google_genai": "google_genai:gemini-3.5-flash-lite",
    # Etiqueta GGUF (-q4_K_M) y no «gemma4:e4b» a secas: en Apple Silicon, la etiqueta por defecto baja la
    # versión MLX, y el runner MLX de Ollama 0.40 no admite salida estructurada (`format`). Ver modelos_locales.py.
    "ollama": "ollama:gemma4:e4b-it-q4_K_M",
    "offline": "offline:extractivo",
}
DEFAULT_EMBEDDINGS = {
    "openai": "openai:text-embedding-3-small",
    "google_genai": "google_genai:gemini-embedding-001",
    "ollama": "ollama:qwen3-embedding:0.6b",
    "offline": "offline:hashing",
    # Anthropic no ofrece embeddings: sin embeddings el RAG usa solo BM25.
    "anthropic": None,
}
API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}


def _resolve(value: str | None, defaults: dict[str, str | None]) -> str | None:
    if not value:
        return None
    if ":" not in value:
        return defaults.get(value)
    return value


def llm_model_name() -> str:
    model = _resolve(os.getenv("LLM_MODEL", "google_genai"), DEFAULT_LLM)
    if model is None:
        raise ValueError("LLM_MODEL no es válido. Revisa tu fichero .env")
    return model


def embeddings_model_name() -> str | None:
    value = os.getenv("EMBEDDINGS_MODEL")
    if value is None:
        provider = llm_model_name().split(":", 1)[0]
        return DEFAULT_EMBEDDINGS.get(provider)
    if value.lower() in {"", "none", "ninguno"}:
        return None
    return _resolve(value, DEFAULT_EMBEDDINGS)


@cache
def get_llm() -> BaseChatModel:
    """Devuelve el modelo de chat configurado en `.env` (LLM_MODEL)."""
    model = llm_model_name()
    if model.startswith("offline:"):
        from taller.offline import OfflineChatModel

        return OfflineChatModel()
    if model.startswith("ollama:"):
        from taller.modelos_locales import ChatOllamaRobusto

        # Salida estructurada con respaldo para las etiquetas MLX, que no admiten `format` (ver modelos_locales.py).
        return ChatOllamaRobusto(model=model.split(":", 1)[1])
    return init_chat_model(model)


@cache
def get_embeddings() -> Embeddings | None:
    """Devuelve el modelo de embeddings, o None si el proveedor no tiene."""
    model = embeddings_model_name()
    if model is None:
        return None
    if model.startswith("offline:"):
        from taller.offline import HashingEmbeddings

        return HashingEmbeddings()
    from taller.cache import EmbeddingsCache

    # Cada texto se vectoriza una vez por máquina: protege la cuota diaria de la capa gratuita.
    return EmbeddingsCache(init_embeddings(model), model, ROOT / ".cache" / "embeddings.sqlite")


def ruta_memoria(nombre: str = "memoria") -> Path:
    """Fichero SQLite del store en `.data/`, uno por modelo de embeddings.

    El índice semántico guarda vectores con las dimensiones del modelo: si alguien cambia de
    proveedor (de Gemini a Ollama, o a offline), un fichero compartido daría `Vector dimension
    mismatch`. Con el nombre del modelo en el fichero, cada proveedor tiene su memoria.
    """
    import re

    modelo = re.sub(r"[^a-z0-9]+", "-", (embeddings_model_name() or "sin-embeddings").lower()).strip("-")
    carpeta = ROOT / ".data"
    carpeta.mkdir(exist_ok=True)
    return carpeta / f"{nombre}-{modelo}.sqlite"


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "no instalado"


# Modelos de Gemini que en la capa gratuita tienen 20 peticiones al día (medido el 4-nov-2026).
_GEMINI_CUOTA_CORTA = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash")


def _pista(exc: Exception, proveedor: str = "google_genai") -> str | None:
    """Traduce los errores más habituales de los proveedores a una acción concreta."""
    texto = str(exc)
    if "API_KEY_INVALID" in texto or "API key not valid" in texto:
        return "La clave no es válida: vuelve a copiarla de aistudio.google.com/apikey y reinicia el kernel."
    if "401" in texto or "Unauthorized" in texto or "authentication" in texto.lower():
        return "El proveedor rechaza la clave: revisa que esté completa y sea del proveedor de LLM_MODEL."
    if "404" in texto and ("not found" in texto.lower() or "NOT_FOUND" in texto):
        return f"Ese modelo no existe con ese nombre. Deja LLM_MODEL={proveedor} (el proveedor a secas)."
    if "429" in texto or "RESOURCE_EXHAUSTED" in texto:
        if (
            "per day" in texto.lower()
            or "limit: 20" in texto
            or "limit: 1000" in texto
            or re.search(r"retry in \d+h", texto)
        ):
            return "Cuota diaria agotada. Usa LLM_MODEL=google_genai (Flash-Lite) u otra clave."
        return "Límite por minuto: espera un minuto y vuelve a ejecutar esta celda."
    if "503" in texto or "UNAVAILABLE" in texto:
        return "El modelo está saturado ahora mismo: espera un poco o cambia de modelo (README → Elige un modelo)."
    if "Connection refused" in texto or "ConnectError" in type(exc).__name__:
        return "No se llega al servidor: si usas Ollama, arranca la aplicación o ejecuta `ollama serve`."
    return None


def _ollama_host() -> str:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def _ollama_runner(modelo: str) -> str | None:
    """Runner con el que Ollama ejecuta el modelo (`llamacpp` o `mlx`), o None si no se puede saber."""
    import httpx

    try:
        detalles = httpx.post(f"{_ollama_host()}/api/show", json={"model": modelo}, timeout=10).json()
        return detalles.get("details", {}).get("runner")
    except Exception:  # noqa: BLE001
        return None


def _ollama_disponible(modelos: list[str]) -> tuple[bool, str]:
    """Comprueba que Ollama responde y que tiene descargados los modelos indicados."""
    import httpx

    host = _ollama_host()
    try:
        nombres = {m["name"] for m in httpx.get(f"{host}/api/tags", timeout=5).json()["models"]}
    except Exception as exc:  # noqa: BLE001
        return (
            False,
            f"Ollama no responde en {host} ({type(exc).__name__}). Abre la app o ejecuta `ollama serve`.",
        )
    faltan = [m for m in modelos if m not in nombres and f"{m}:latest" not in nombres]
    if faltan:
        return False, "Faltan modelos en Ollama: " + " · ".join(f"ollama pull {m}" for m in faltan)
    return True, f"Ollama responde en {host} con {', '.join(modelos)}"


_CREAR_CUENTA = (
    "Para tenerla: cuenta gratuita en eu.smith.langchain.com (Sign up con Google, GitHub o correo; sin tarjeta), "
    "Settings → API Keys → Create API Key, pégala en la línea LANGSMITH_API_KEY= de .env y reinicia el kernel"
)


def _comprobar_langsmith(ok, aviso, error, proveedor: str) -> None:
    """Studio inicia sesión en eu.smith.langchain.com con la cuenta de LangSmith. Cuenta y clave son opcionales: solo avisa."""
    import httpx

    clave = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not clave:
        aviso(
            "sin LANGSMITH_API_KEY: opcional; Studio te pedirá iniciar sesión y el 05 no tendrá trazas. "
            + _CREAR_CUENTA
        )
        return
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com").rstrip("/")
    try:
        estado = httpx.get(
            f"{endpoint}/api/v1/sessions", params={"limit": 1}, headers={"x-api-key": clave}, timeout=10
        ).status_code
    except Exception as exc:  # noqa: BLE001
        aviso(f"no se ha podido comprobar la clave de LangSmith ({type(exc).__name__}); Studio necesita red")
        return
    if estado == 200:
        ok("cuenta de LangSmith: la clave es válida; Studio abrirá con esa cuenta (Chrome o Firefox)")
    elif estado in (401, 403):
        aviso(
            f"LangSmith rechaza la clave (HTTP {estado}); opcional, pero si la quieres vuelve a copiarla de "
            "eu.smith.langchain.com → Settings → API Keys (cuenta en EE. UU.: LANGSMITH_ENDPOINT=https://api.smith.langchain.com)"
        )
    else:
        aviso(
            f"LangSmith responde HTTP {estado} al comprobar la clave; si Studio no inicia sesión, revisa la clave"
        )


def check_env(call_models: bool = True) -> bool:
    """Comprueba que el taller entero puede ejecutarse en esta máquina.

    Python y paquetes, `.env`, clave y modelo, que el LLM responde, que sabe pedir herramientas
    y devolver salida estructurada, los embeddings, la base de conocimiento, el checkpointer
    SQLite y, con aviso si falta, la clave de LangSmith para Studio. Si termina en «Todo listo.», todos los notebooks funcionan.
    Con `call_models=False` solo comprueba Python, paquetes y `.env` (sin red).
    """
    import time

    errores: list[str] = []
    avisos: list[str] = []

    def ok(msg: str) -> None:
        print(f"  OK     {msg}")

    def error(msg: str, pista: str | None = None) -> None:
        print(f"  ERROR  {msg}")
        if pista:
            print(f"         → {pista}")
        errores.append(msg)

    def aviso(msg: str) -> None:
        print(f"  AVISO  {msg}")
        avisos.append(msg)

    def cronometrado(fn):
        t0 = time.perf_counter()
        resultado = fn()
        segundos = time.perf_counter() - t0
        if segundos > 20:
            aviso(
                f"la llamada ha tardado {segundos:.0f} s: el proveedor va lento ahora; no reejecutes la celda, espera"
            )
        return resultado, segundos

    # 1. Python y paquetes
    print("Python y paquetes")
    py = platform.python_version()
    if tuple(int(x) for x in py.split(".")[:2]) >= (3, 12):
        ok(f"Python {py}")
    else:
        error(
            f"Python {py}: el taller necesita 3.12 o superior",
            "uv sync lo instala solo; sin uv, instala Python 3.12 a 3.14",
        )
    for pkg in ("langgraph", "langchain", "langchain-core", "langgraph-checkpoint-sqlite"):
        v = _pkg(pkg)
        (ok if v != "no instalado" else error)(f"{pkg} {v}")
    cli = _pkg("langgraph-cli")

    # 2. .env
    print("\nConfiguración (.env)")
    env_path = ROOT / ".env"
    if env_path.exists():
        ok(f"{env_path.name} encontrado")
    elif (ROOT / ".env.txt").exists():
        error(
            "existe .env.txt, pero el fichero tiene que llamarse .env (sin .txt)",
            "renómbralo: en Windows, desde PowerShell: Rename-Item .env.txt .env",
        )
    else:
        error(
            "no existe el fichero .env",
            "cp .env.example .env (en Windows: copy .env.example .env) y pon tu clave",
        )
    try:
        llm, emb = llm_model_name(), embeddings_model_name()
    except ValueError as exc:
        error(
            str(exc),
            "LLM_MODEL tiene que ser google_genai, openai, anthropic, ollama, offline o proveedor:modelo",
        )
        print("\nRevisa los errores de arriba (README → Problemas frecuentes).")
        return False
    proveedor = llm.split(":", 1)[0]
    ok(f"LLM_MODEL = {llm}")
    if emb:
        ok(f"EMBEDDINGS_MODEL = {emb}")
    else:
        aviso("sin embeddings: el RAG del bloque 4 usará solo BM25 (búsqueda léxica)")
    if llm.split(":", 1)[1] in _GEMINI_CUOTA_CORTA:
        aviso(
            "ese modelo de Gemini solo da 20 peticiones al día gratis y se agota en el notebook 02; usa google_genai (Flash-Lite)"
        )
    for prov in {proveedor, (emb or "").split(":")[0]}:
        key = API_KEYS.get(prov)
        if key and not os.getenv(key):
            error(f"falta {key} en .env", f"pon la clave en la línea {key}=... de .env y reinicia el kernel")
    if proveedor == "offline":
        aviso(
            "modo offline: el «modelo» es un imitador con reglas; la mecánica funciona, las respuestas no son reales"
        )
    if proveedor == "ollama" or (emb or "").startswith("ollama:"):
        modelos = [m.split(":", 1)[1] for m in (llm, emb or "") if m.startswith("ollama:")]
        disponible, detalle = _ollama_disponible(modelos)
        (ok if disponible else error)(detalle)
        if disponible and proveedor == "ollama" and _ollama_runner(llm.split(":", 1)[1]) == "mlx":
            aviso(
                "ese modelo corre en el runner MLX de Ollama, que no admite salida estructurada nativa (format); "
                "el taller la suple pidiendo el JSON por instrucciones. Para la nativa, usa la etiqueta GGUF del README: "
                f"ollama pull {DEFAULT_LLM['ollama'].split(':', 1)[1]}"
            )

    if errores or not call_models:
        print(
            "\n"
            + (
                "Revisa los errores de arriba (README → Problemas frecuentes)."
                if errores
                else "Todo listo (sin llamar a los modelos)."
            )
        )
        return not errores

    # 3. El modelo
    print("\nModelo de chat")
    try:
        respuesta, s = cronometrado(
            lambda: get_llm().invoke("Responde solo con la palabra: listo").text.strip()
        )
        if "listo" in respuesta.lower():
            ok(f"responde ({s:.1f} s)")
        else:
            aviso(
                f"responde, pero no lo que se le pide ({respuesta[:60]!r}); suele ser normal en modelos pequeños"
            )
    except Exception as exc:  # noqa: BLE001 — queremos mostrar cualquier fallo
        error(f"el modelo no responde → {exc}", _pista(exc, proveedor))
        print("\nRevisa los errores de arriba (README → Problemas frecuentes).")
        return False

    from langchain_core.tools import tool
    from pydantic import BaseModel, Field

    @tool
    def sumar(a: int, b: int) -> int:
        """Suma dos números enteros."""
        return a + b

    class Capital(BaseModel):
        ciudad: str = Field(description="Nombre de la ciudad")
        es_capital: bool = Field(description="True si es la capital del país")

    try:
        msg, s = cronometrado(
            lambda: get_llm().bind_tools([sumar]).invoke("¿Cuánto es 2 + 3? Usa la herramienta sumar.")
        )
        if msg.tool_calls:
            ok(
                f"sabe pedir herramientas: {msg.tool_calls[0]['name']}({msg.tool_calls[0]['args']}) ({s:.1f} s)"
            )
        else:
            aviso(
                "ha respondido sin usar la herramienta; en los bloques 3 a 6 quizá tengas que pedir las cosas más explícitamente"
            )
    except Exception as exc:  # noqa: BLE001
        error(
            f"no admite herramientas → {exc}",
            "el modelo tiene que soportar tool calling (los del README lo hacen)",
        )
    try:
        dato, s = cronometrado(
            lambda: get_llm().with_structured_output(Capital).invoke("¿Es Madrid la capital de España?")
        )
        if isinstance(dato, Capital):
            ok(f"salida estructurada: {dato.model_dump()} ({s:.1f} s)")
        else:
            aviso(f"salida estructurada devuelve {type(dato).__name__}; el bloque 4 la necesita como objeto")
    except Exception as exc:  # noqa: BLE001
        error(f"salida estructurada falla → {exc}", "el bloque 4 (evaluador del RAG) la necesita")

    # 4. Embeddings y base de conocimiento
    print("\nRecuperación (bloque 4)")
    if emb:
        try:
            dims, s = cronometrado(lambda: len(get_embeddings().embed_query("hola")))
            ok(f"embeddings de {dims} dimensiones ({s:.1f} s)")
        except Exception as exc:  # noqa: BLE001
            error(
                f"los embeddings fallan → {exc}",
                _pista(exc, proveedor) or "EMBEDDINGS_MODEL=none en .env sigue con BM25",
            )
    from taller.kb import load_knowledge_base

    docs = load_knowledge_base()
    (ok if docs else error)(f"base de conocimiento: {len(docs)} fragmentos en datos/conocimiento/")

    # 5. Grafo con checkpointer SQLite (bloque 6)
    print("\nLangGraph")
    try:
        from typing import TypedDict

        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.graph import END, START, StateGraph

        class _Estado(TypedDict):
            n: int

        with SqliteSaver.from_conn_string(":memory:") as saver:
            g = StateGraph(_Estado).add_node("sumar_uno", lambda s: {"n": s["n"] + 1})
            g = g.add_edge(START, "sumar_uno").add_edge("sumar_uno", END).compile(checkpointer=saver)
            cfg = {"configurable": {"thread_id": "check"}}
            final = g.invoke({"n": 1}, cfg)["n"]
            guardado = g.get_state(cfg).values["n"]
        if final == guardado == 2:
            ok("grafo con checkpointer SQLite: ejecuta y guarda el estado")
        else:
            error(f"el grafo devuelve {final} y guarda {guardado}; se esperaba 2")
    except Exception as exc:  # noqa: BLE001
        error(f"el grafo de prueba falla → {exc}")

    # 6. Studio: servidor local y cuenta de LangSmith
    print("\nStudio (bloques 3 a 6; cuenta y clave opcionales)")
    if cli == "no instalado":
        error("langgraph-cli no instalado: `uv run langgraph dev` no existe", "ejecuta `uv sync` otra vez")
    elif not (ROOT / "langgraph.json").exists():
        error("falta langgraph.json en la raíz del repositorio: `uv run langgraph dev` no arrancará")
    else:
        ok(f"langgraph-cli {cli} y langgraph.json: `uv run langgraph dev` arranca el servidor local")
    _comprobar_langsmith(ok, aviso, error, proveedor)

    if errores:
        print("\nRevisa los errores de arriba (README → Problemas frecuentes).")
    elif avisos:
        print(
            f"\nTodo listo. ({len(avisos)} aviso{'s' if len(avisos) > 1 else ''} arriba; no bloquea{'n' if len(avisos) > 1 else ''} el taller.)"
        )
    else:
        print("\nTodo listo.")
    return not errores
