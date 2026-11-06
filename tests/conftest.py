"""Los tests corren siempre en modo offline: sin red, sin claves y en segundos.

Se fijan las variables aquí, antes de importar `taller`, para que el `.env` de quien
desarrolla (proveedor real, trazas) no afecte a los tests.
"""

import os

os.environ["LLM_MODEL"] = "offline"
os.environ["EMBEDDINGS_MODEL"] = "offline"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = ""  # sin red: la comprobación de Studio queda en AVISO
os.environ.pop("LANGSMITH_ENDPOINT", None)
for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.pop(var, None)
