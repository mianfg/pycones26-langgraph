# Introducción a LangGraph: construyendo agentes que aprenden

Taller práctico de 90 minutos para **PyConES 2026** (sábado 7 de noviembre, 18:00, aula 4.1), impartido por Miguel Ángel Fernández Gutiérrez. En español, con tu propio portátil.

Partimos de un grafo de tres líneas y terminamos con el asistente interno de **Cuac & Co.**, una tienda online de patitos de goma coleccionables (inventada): un agente que consulta la documentación de la empresa, decide cuándo necesita buscar, **aprende hechos nuevos mientras lo usas** y los recuerda en conversaciones futuras, con una persona aprobando lo que guarda. Todo con Python, [LangGraph](https://docs.langchain.com/oss/python/langgraph/) y el proveedor de modelos que prefieras.

Si tienes diez minutos antes del taller, haz solo la sección [«Empieza aquí»](#empieza-aquí). El resto de este documento explica el porqué de cada pieza.

## Qué vas a construir

Cada bloque del taller añade una pieza al agente, y cada pieza es un notebook (los bloques 0 y 1, entorno y motivación, no tienen notebook):

| Bloque | Notebook | Lo que añade | Lo que aprendes |
|---|---|---|---|
| 2 | `01_primer_contacto` | Un grafo que clasifica incidencias, sin ningún modelo de lenguaje | Estado, nodos, aristas condicionales, reducers y ciclos |
| 3 | `02_primer_agente` | Un agente que decide qué herramientas usar y vuelve a pensar con el resultado | Herramientas, el bucle del agente, streaming, `create_agent` |
| 4 | `03_rag` | Búsqueda en la documentación de Cuac & Co., integrada en el grafo | BM25, embeddings, fusión RRF y RAG correctivo (evaluar, reformular, reintentar) |
| 5 | `04_memoria_aprendizaje` | Memoria a largo plazo: el agente guarda hechos y preferencias y los reutiliza | Checkpointer (hilo) frente a *store* (largo plazo), `ToolRuntime` |
| 6 | `05_persistencia_produccion` | Todo lo anterior en disco, con aprobación humana y servido por HTTP | SQLite, `interrupt()`, historial, Agent Server, trazas |

El agente final, en una frase: un bucle «modelo → herramientas → modelo» donde una de las herramientas busca en la documentación (y en lo aprendido), otra guarda conocimiento nuevo (tras pedir permiso) y otra guarda preferencias del usuario; LangGraph guarda el estado después de cada paso, así que el agente puede pausarse, reanudarse y sobrevivir a un reinicio.

```
                 ┌──────────────┐
    usuario ───▶ │    agent     │ ───▶ respuesta
                 │ (LLM + prompt│
                 │  + prefs)    │
                 └──────┬───────┘
                        │ pide herramientas
                        ▼
                 ┌──────────────┐   buscar_documentacion  → BM25 + embeddings + lo aprendido
                 │    tools     │   aprender_hecho        → interrupt() → store ("conocimiento")
                 │              │   guardar_preferencia   → store ("usuarios", user_id)
                 └──────┬───────┘   hora_actual, dias_laborables
                        └────────▶ vuelve a agent
```

**Cada notebook se puede empezar sin haber hecho los anteriores.** Lo construido en bloques previos se importa ya resuelto del paquete `taller/` (por ejemplo, el notebook 04 obtiene la recuperación del 03 con `build_retriever()`). Si te pierdes en un bloque, abre el notebook del siguiente y ejecútalo desde la primera celda. Los ejercicios tienen la solución plegada debajo («Ver solución») y cada notebook termina con un reto opcional.

## Empieza aquí

Cuatro pasos (el tercero es opcional); si el cuarto acaba en **«Todo listo.»**, has terminado.

**1. Instala [uv](https://docs.astral.sh/uv/)** (gestiona Python y las dependencias; no toca el Python que ya tengas). Una de estas líneas, en una terminal (en Windows, PowerShell):

| Sistema | Comando |
|---|---|
| macOS o Linux | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| macOS con Homebrew | `brew install uv` |
| Windows (PowerShell) | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| Windows con winget | `winget install --id=astral-sh.uv -e` |
| Cualquiera, si ya tienes Python | `pip install uv` (o `pipx install uv`) |

Cierra y vuelve a abrir la terminal: `uv --version` tiene que responder. Si ya lo tenías, `uv self update`. ¿No quieres uv? Más abajo está la [ruta con pip](#sin-uv-entorno-virtual-y-pip).

**2. Clona, instala y pon tu clave:**

```bash
git clone https://github.com/mianfg/pycones26-langgraph
cd pycones26-langgraph
uv sync                   # descarga Python 3.14 y las dependencias exactas de uv.lock
cp .env.example .env      # en Windows: copy .env.example .env
```

Ahora elige **una** de las dos opciones y edita `.env`. Es un fichero oculto (empieza por punto): ábrelo desde el panel de ficheros de JupyterLab, que en este proyecto los muestra, o con tu editor (`code .env`, en Mac `open -e .env`, en Windows `notepad .env`):

- **Opción A · Google AI Studio** (recomendada: gratis, sin tarjeta, dos minutos). Entra en [aistudio.google.com/apikey](https://aistudio.google.com/apikey), pulsa «Create API key», copia la clave y pégala en la línea `GOOGLE_API_KEY=` de `.env`. `LLM_MODEL=google_genai` ya viene puesto.
- **Opción B · Ollama** (modelos en tu portátil, sin clave ni cuotas; 7 GB de descarga, hazla en casa). Instala [Ollama](https://ollama.com), ejecuta `ollama pull gemma4:e4b-it-q4_K_M` y `ollama pull qwen3-embedding:0.6b`, y en `.env` pon `LLM_MODEL=ollama`. Con 8 GB de memoria, mira la tabla de [«Elige un modelo»](#elige-un-modelo).

Otras opciones (OpenAI, Anthropic, o un modo sin red ni clave) en [«Elige un modelo»](#elige-un-modelo).

**3. Cuenta para Studio (opcional, recomendado).** Studio es la interfaz que dibuja el grafo y lo ilumina mientras se ejecuta; la usamos en los bloques 3 a 6 y vive en [eu.smith.langchain.com](https://eu.smith.langchain.com), así que para abrirla en tu portátil hace falta **iniciar sesión con una cuenta de LangSmith**: *Sign up* (Google, GitHub o correo; gratis, sin tarjeta). Si además generas una clave en *Settings → API Keys → Create API Key* y la pegas en la línea `LANGSMITH_API_KEY=` de `.env`, el notebook 00 la comprueba y el notebook 05 podrá enseñar trazas; sin ella solo recibirás un aviso. Sin cuenta, el taller se sigue igual desde los notebooks y ves Studio en la pantalla del ponente. El notebook 00 termina abriendo Studio una vez (`uv run studio` en otra terminal, con Chrome o Firefox). Detalles en [«Studio: ver el grafo en directo»](#studio-ver-el-grafo-en-directo).

**4. Comprueba:**

```bash
uv run jupyter lab
```

Abre `notebooks/00_preparacion.ipynb` y ejecútalo entero. Comprueba todo lo que usa el taller: Python y paquetes, `.env` y la clave (o que Ollama responde y tiene los modelos), que el modelo responde, que sabe pedir herramientas y devolver salida estructurada, los embeddings, la base de conocimiento, un grafo con checkpointer SQLite, que `langgraph dev` puede arrancar y, si la has puesto, que la clave de LangSmith es válida (Studio). Cada línea sale como `OK`, `AVISO` (no bloquea) o `ERROR` (con qué hacer debajo). Si termina en **«Todo listo.»**, todos los notebooks van a funcionar. Lo mismo desde la terminal:

```bash
uv run python -c "from taller.config import check_env; check_env()"
```

## Elige un modelo

Cada asistente usa **su propia clave**; el taller no proporciona ninguna. El proveedor se elige en `.env` con `LLM_MODEL` (y, opcionalmente, `EMBEDDINGS_MODEL`); todo el código pasa por `get_llm()` y `get_embeddings()` de `taller/config.py`, así que cambiar de proveedor es cambiar una línea.

| `LLM_MODEL` | Modelo por defecto | Embeddings por defecto | Clave | Cuándo |
|---|---|---|---|---|
| `google_genai` | `gemini-3.5-flash-lite` | `gemini-embedding-001` | `GOOGLE_API_KEY` (gratis) | **Recomendado**: gratis, sin tarjeta, rápido |
| `openai` | `gpt-6-luna` | `text-embedding-3-small` | `OPENAI_API_KEY` | Si ya tienes clave; el taller cuesta céntimos |
| `anthropic` | `claude-haiku-5-5` | ninguno: el RAG usa solo BM25 | `ANTHROPIC_API_KEY` | Si ya tienes clave |
| `ollama` | `gemma4:e4b-it-q4_K_M` | `qwen3-embedding:0.6b` | ninguna | Modelos locales, sin cuotas; descarga previa de 2,5 a 7 GB |
| `offline` | imitador basado en reglas | hashing léxico | ninguna | Plan B sin red ni clave |

Cualquier otro modelo vale con el formato `proveedor:modelo` (`LLM_MODEL=openai:gpt-6.1-sol`), y cualquier servicio compatible con la API de OpenAI (LiteLLM, vLLM, un proxy) con `LLM_MODEL=openai:<modelo>` y `OPENAI_BASE_URL`.

**Gemini y la capa gratuita.** El taller entero gasta unas 60 llamadas al modelo y, gracias a la caché de embeddings de `.cache/`, unos 60 textos vectorizados: muy por debajo de los límites diarios de `gemini-3.5-flash-lite` (varios cientos) y de los embeddings (1 000 textos). Por eso el modelo por defecto no es `gemini-3.8-flash`: en la capa gratuita, los Flash grandes (3.8, 3.7 y 3.6) tienen **20 peticiones al día**, que se agotan en el segundo notebook, y el 4 de noviembre de 2026 además devolvían 503 por saturación. Si pagas, cambia a `google_genai:gemini-3.8-flash`. Google puede usar las peticiones gratuitas para mejorar sus productos: aquí solo se envían documentos inventados.

**Ollama (local).** Instala [Ollama](https://ollama.com) y descarga los modelos **antes del taller, con buena conexión**. Según la memoria del portátil:

| Portátil | Descarga | `.env` |
|---|---|---|
| 16 GB o más | `ollama pull gemma4:e4b-it-q4_K_M` y `ollama pull qwen3-embedding:0.6b` (7,2 GB) | `LLM_MODEL=ollama` |
| 8 GB | `ollama pull qwen3.5:4b-q4_K_M` y `ollama pull embeddinggemma` (3,9 GB) | `LLM_MODEL=ollama:qwen3.5:4b-q4_K_M` y `EMBEDDINGS_MODEL=ollama:embeddinggemma` |
| Lo mínimo | `ollama pull qwen3.5:2b-q4_K_M` y `ollama pull embeddinggemma` (2,5 GB) | `LLM_MODEL=ollama:qwen3.5:2b-q4_K_M` y `EMBEDDINGS_MODEL=ollama:embeddinggemma` |

Los tres admiten llamadas a herramientas y salida estructurada. **Usa las etiquetas largas (`-q4_K_M`, formato GGUF)**: en Apple Silicon, las cortas (`gemma4:e4b`, `qwen3.5:4b`) descargan la versión MLX del modelo, y el runner MLX de Ollama 0.40 rechaza la salida estructurada (`structured output is unavailable`, error 501), que usa el evaluador del RAG; si ya la tienes, `taller/modelos_locales.py` la suple pidiendo el JSON por instrucciones, y el notebook 00 lo avisa. Probado el 5 de noviembre de 2026 en un Apple Silicon con Ollama 0.40.14: `gemma4:e4b-it-q4_K_M` pasa todo el taller con tiempos parecidos a Gemini (notebook 02 en 45 s, 03 en 56 s, 04 en 63 s); `qwen3.5:4b-q4_K_M` también pasa, pero «piensa» antes de responder y el notebook 03 le lleva ocho minutos y medio (el 02, 76 s): quien venga con 8 GB irá por detrás del grupo en el bloque 4. Cuanto más pequeño el modelo, peor decide cuándo usar una herramienta: en los bloques 4 y 5 quizá tengas que pedir las cosas de forma más explícita. Para embeddings, `embeddinggemma` (622 MB) y `qwen3-embedding:0.6b` (639 MB) son multilingües; evita `nomic-embed-text`, entrenado en inglés.

**Modo `offline`.** Si falla la wifi o tu clave, `LLM_MODEL=offline`. Los grafos, las herramientas, la recuperación BM25, la memoria y la persistencia funcionan de verdad; solo el «modelo» es un imitador que elige herramientas por palabras clave y responde extrayendo frases.

## Cómo seguir el taller

Arranca JupyterLab desde la raíz del repositorio (`uv run jupyter lab`) y abre los notebooks de `notebooks/` en orden. Cada uno indica al principio qué necesita (normalmente solo tu `.env`), qué objetivo tiene y dónde están los ejercicios. Las celdas con `# TODO` son los ejercicios; justo debajo, plegada, está la solución.

La base de conocimiento que consulta el agente está en `datos/conocimiento/`: diez documentos en Markdown que simulan la wiki interna de Cuac & Co., una tienda online de patitos de goma coleccionables (arquitectura, despliegues, guardias, errores conocidos, vacaciones...). Como el modelo no puede conocer esa información, cualquier respuesta correcta tiene que salir de la recuperación. Puedes leerlos para saber qué preguntar.

Después del taller, el mejor ejercicio es cambiar la base de conocimiento por documentos tuyos: cada fichero es un tema y cada sección `##` se convierte en un fragmento recuperable.

## Studio: ver el grafo en directo

[LangSmith Studio](https://docs.langchain.com/langsmith/studio) dibuja el grafo, lo ilumina nodo a nodo, enseña cada llamada a herramienta y permite aprobar las pausas humanas desde un formulario. Lo usamos del bloque 3 al 6. La interfaz vive en eu.smith.langchain.com e inicia sesión con una cuenta de LangSmith; el servidor corre en tu portátil y arranca sin clave. Cuenta y clave son opcionales: sin cuenta, todo el taller se sigue desde los notebooks y ves Studio en la pantalla del ponente; sin clave, el notebook 00 solo avisa y el 05 no enseña trazas.

1. Cuenta gratuita en [eu.smith.langchain.com](https://eu.smith.langchain.com): *Sign up* con Google, GitHub o correo (plan Developer, sin tarjeta). Genera una clave en *Settings → API Keys → Create API Key* y ponla en `LANGSMITH_API_KEY` en `.env` (cuenta en la región europea: además `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`). Con `LANGSMITH_TRACING=false` (el valor por defecto) nada sale de tu máquina: la clave solo sirve para iniciar sesión en la interfaz y el servidor local arranca igual sin ella.
2. Arranca el servidor local desde la raíz del repositorio, en otra terminal:

   ```bash
   uv run studio
   ```

   `uv run studio` es un atajo de `uv run langgraph dev --studio-url https://eu.smith.langchain.com` (está en `taller/cli.py`): arranca el Agent Server local y abre Studio en la región europea de LangSmith; sin `--studio-url`, `langgraph dev` abre la de EE. UU. mientras las trazas estén desactivadas. Admite los mismos argumentos (`uv run studio --port 2025`).

3. Se abre Studio solo; si no, entra en <https://eu.smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024>. Usa Chrome o Firefox. Chrome 142 o superior pide permiso de «Local network access» la primera vez: acéptalo. Safari bloquea la conexión con el servidor local (alternativa: `uv run langgraph dev --tunnel`).

El servidor expone los tres grafos declarados en `langgraph.json` (`agente_basico`, `rag` y `agente_que_aprende`, definidos en `taller/studio.py`) y se recarga al guardar cambios en `taller/`. Es el mismo servidor que se usaría en producción (Agent Server), en su versión de desarrollo: guarda hilos, asistentes y memoria en `.langgraph_api/`.

Para ver qué hace el agente por dentro (prompts, herramientas, tokens, latencia): `LANGSMITH_TRACING=true` con la misma clave, o [Langfuse](https://langfuse.com) (código abierto, autoalojable) con `LANGFUSE_PUBLIC_KEY` y `LANGFUSE_SECRET_KEY`. Lo enseña la sección 5.6 del notebook 05.

## Demo del agente final

Sin notebook, en la terminal:

```bash
uv run python -m taller.demo --usuario ana
```

Pregúntale por Cuac & Co. («¿Puedo desplegar un viernes?»), cuéntale un cambio («Apunta esto: ...») y aprueba o rechaza lo que quiere guardar; después, `/nuevo` abre otra conversación y verás que lo recuerda. `/memoria` enseña lo aprendido y `/salir` termina. Los datos se guardan en `.data/`; bórrala para empezar de cero.

## Mapa del repositorio

```
notebooks/            los seis notebooks del taller, en orden (00 comprueba el entorno)
taller/               el código del taller como paquete: lo que importan los notebooks
  config.py             get_llm() y get_embeddings() según .env; check_env()
  modelos_locales.py    Ollama con salida estructurada de respaldo (las etiquetas MLX no admiten `format`)
  kb.py                 carga y trocea la base de conocimiento (Markdown → fragmentos)
  retrieval.py          BM25, índice de embeddings, Reciprocal Rank Fusion, HybridRetriever
  tools.py              herramientas del agente (las de memoria usan ToolRuntime e interrupt)
  agents.py             el grafo de cada bloque, ya resuelto: los «puntos de recuperación»
  studio.py             los grafos tal y como los sirve langgraph dev (langgraph.json apunta aquí)
  cli.py                `uv run studio`: atajo de `langgraph dev --studio-url https://eu.smith.langchain.com`
  servidor.py           cliente del servidor local para los notebooks (SDK de LangGraph)
  demo.py               demo de terminal del agente final
  cache.py              caché en disco de embeddings (.cache/): cada texto se vectoriza una vez
  observabilidad.py     trazas opcionales con LangSmith o Langfuse
  offline.py            modelos de juguete para LLM_MODEL=offline
  viz.py                mostrar_grafo() y mostrar_mensajes()
datos/conocimiento/   la wiki interna de Cuac & Co. (ficticia), diez ficheros Markdown
tests/                tests en modo offline, sin red ni claves (uv run pytest)
scripts/              comprobar_notebooks.py (ejecuta los seis) y probar_servidor.py (recorre el servidor por HTTP)
extras/               configuración de Jupyter que uv sync instala en el .venv: JupyterLab muestra los ficheros ocultos (.env)
langgraph.json        qué grafos sirve el Agent Server y cómo indexa su memoria
pyproject.toml        dependencias; uv.lock fija las versiones; requirements.txt es su copia para pip
.env.example          plantilla de configuración: cópiala a .env
```

Carpetas que aparecen al usarlo, ignoradas por git: `.venv/` (el entorno), `.data/` (SQLite de la demo y del notebook 05), `.cache/` (embeddings ya calculados: no la borres, ahorra cuota), `.langgraph_api/` (estado del servidor local).

## Sin uv: entorno virtual y pip

Necesitas **Python 3.12 a 3.14** ya instalado (`python --version`; en macOS y Linux quizá sea `python3`). `requirements.txt` lleva las mismas versiones que `uv.lock`.

```bash
git clone https://github.com/mianfg/pycones26-langgraph
cd pycones26-langgraph
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -e .
cp .env.example .env               # y edita .env con tu clave
jupyter lab --ContentsManager.allow_hidden=True   # con pip no se instala la configuración de extras/: así JupyterLab deja ver .env (View → Show Hidden Files)
```

A partir de ahí, cada `uv run <comando>` de este documento es simplemente `<comando>` con el entorno activado (`jupyter lab`, `pytest`, `langgraph dev`, `python -m taller.demo`). Con conda es lo mismo: `conda create -n taller python=3.14`, `conda activate taller` y los mismos `pip install`.

## Comprobaciones

```bash
uv run pytest                                            # ~2 s, sin red ni claves
uv run python scripts/comprobar_notebooks.py             # ejecuta los seis notebooks con tu .env
LLM_MODEL=offline uv run python scripts/comprobar_notebooks.py   # en PowerShell: $env:LLM_MODEL="offline"; uv run python scripts/comprobar_notebooks.py
uv run python scripts/probar_servidor.py                 # con el servidor local arrancado en otra terminal
```

El último recorre por HTTP lo mismo que haremos en Studio: una herramienta, el RAG, una aprobación humana (interrupción y reanudación) y la memoria entre hilos. Los notebooks ejecutados quedan en `.data/notebooks_ejecutados/`, útiles como resultados de respaldo si falla la red.

## Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `uv: command not found` | Abre una terminal nueva tras instalar uv. Si sigue sin aparecer, `pip install uv` o la ruta sin uv |
| `uv sync` falla al descargar Python | Sin red o proxy corporativo: prueba `uv sync --native-tls`; si no, instala Python 3.12+ a mano y sigue la ruta sin uv |
| El grafo sale como texto, no como dibujo | Usa JupyterLab (`uv run jupyter lab`) o `mostrar_grafo(g, png=True)` (necesita internet) |
| `Error: falta GOOGLE_API_KEY` | Revisa `.env`: el nombre del fichero (no `.env.txt`), el de la variable y que tenga valor |
| `LLM_MODEL no es válido` | El proveedor tiene que ser `google_genai`, `openai`, `anthropic`, `ollama` u `offline`, o `proveedor:modelo` |
| `400 ... API key not valid` | Clave mal copiada o revocada: vuelve a copiarla y reinicia el kernel |
| `404 ... models/<nombre> is not found` | Ese modelo no existe; usa `gemini-3.5-flash-lite` o consulta la lista en AI Studio |
| Error 429 «per minute» | Capa gratuita de Gemini: espera un minuto (embeddings: 100 textos por minuto) |
| Error 429 «per day» del modelo | Cuota diaria agotada (20 con `gemini-3.8-flash`): vuelve a `gemini-3.5-flash-lite` o a `offline` y reinicia el kernel |
| Error 429 «per day» de embeddings | 1 000 textos al día. La caché de `.cache/` evita repetirlos: no la borres. Si aun así se agota, `EMBEDDINGS_MODEL=none` (el RAG sigue con BM25) |
| Error 503 «high demand» | Modelo saturado: cambia de modelo (`google_genai:gemini-3.1-flash-lite`, u `offline`) y reinicia el kernel |
| La celda tarda más de un minuto | Latencia de la capa gratuita a algunas horas; no reejecutes (cada clic es otra llamada) |
| `Connection refused` con Ollama / `model '...' not found` | Arranca Ollama (`ollama serve`) / descarga el modelo (`ollama pull ...`) |
| He cambiado `.env` y no hace caso | Reinicia el kernel: la configuración se lee al importar `taller` |
| El agente no llama a una herramienta | Reformula de forma más explícita («Apunta esto: ...»); los modelos pequeños lo necesitan a veces |
| `ModuleNotFoundError: taller` | Ejecuta Jupyter con `uv run` desde la raíz del repositorio (ruta sin uv: no olvides el `-e .`) |
| Studio dice «Failed to initialize» | El servidor no está arrancado, o el navegador bloquea `localhost`: Chrome 142+, permite «Local network access»; Safari, `--tunnel`; Brave, desactiva Shields; prueba también `baseUrl=http://localhost:2024` |
| `Port 2024 is already in use, using port ... instead` | Hay otro `langgraph dev` abierto: ciérralo y vuelve a lanzar, o los notebooks seguirán hablando con el viejo |
| `langgraph dev` no arranca | Lánzalo desde la raíz, con `.env` creado. Si acaba en `429 ... embed_content ... per day`, pon `EMBEDDINGS_MODEL=none`. Si se queja de una llamada bloqueante, añade `--allow-blocking` |
| Studio se queda cargando tras editar `taller/studio.py` | El fichero tiene un error de sintaxis (míralo en la terminal del servidor): corrígelo y guarda; el servidor se recupera solo |
| Cambio `LLM_MODEL` en la terminal y el servidor no hace caso | `.env` manda sobre las variables de entorno en `langgraph dev`: edita `.env` y relanza |
| `Vector dimension mismatch` en el notebook 05 | Un fichero de memoria en `.data/` con vectores de otro proveedor (desde el 5 de noviembre cada modelo de embeddings usa su propio fichero, así que solo pasa con un `.data/` anterior): borra `.data/` y vuelve a ejecutar |

## Versiones

Probado con Python 3.14 y:

| Paquete | Versión |
|---|---|
| langgraph | 1.2.19 |
| langchain | 1.4.8 |
| langchain-core | 1.7.4 |
| langgraph-checkpoint-sqlite | 3.1.3 |
| langgraph-cli[inmem] (Agent Server local) | 0.4.40 (langgraph-api 0.17.1) |
| langgraph-sdk | 0.4.12 |
| langsmith / langfuse | 0.14.17 / 5.1.0 |
| langchain-openai / -anthropic / -google-genai / -ollama | 1.7.3 / 1.7.9 / 4.4.2 / 1.1.0 |
| jupyterlab | 4.6.5 |

`uv.lock` fija exactamente estas versiones; `requirements.txt` es su exportación para pip. Si cambias dependencias: `uv export --format requirements-txt --no-hashes --all-groups --no-emit-project -o requirements.txt`.

## Créditos

Cuac & Co., sus productos y sus herramientas internas son inventados para este taller. Cualquier parecido con una empresa real es pura coincidencia.
