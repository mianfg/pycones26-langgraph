"""Ejecuta todos los notebooks de principio a fin y falla si alguno da error.

    uv run python scripts/comprobar_notebooks.py            # con el modelo de tu .env
    LLM_MODEL=offline uv run python scripts/comprobar_notebooks.py

Rellena el único ejercicio cuyo hueco dejaría sin efecto el resto del notebook (`SOLUCIONES`:
sin él, el agente del 04 no aprendería nada) antes de ejecutar; los demás huecos (`...`) no fallan. Los notebooks ejecutados se guardan en .data/notebooks_ejecutados/
(útiles como «resultados guardados» por si falla la red durante el taller).
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
OUT = ROOT / ".data" / "notebooks_ejecutados"

# Ejercicios que hay que completar para que el notebook corra entero: línea `TODO` → solución.
# Se sustituye la línea del TODO y el `...` que la sigue.
SOLUCIONES = {
    "# TODO: guarda `value` en runtime.store": "runtime.store.put(KNOWLEDGE_NS, uuid.uuid4().hex, value)",
}


def rellenar(nb) -> int:
    """Aplica SOLUCIONES a las celdas de código. Devuelve cuántas sustituciones ha hecho."""
    hechas = 0
    for celda in nb.cells:
        if celda.cell_type != "code":
            continue
        lineas = celda.source.split("\n")
        for i, linea in enumerate(lineas):
            for todo, solucion in SOLUCIONES.items():
                if todo in linea:
                    sangria = linea[: len(linea) - len(linea.lstrip())]
                    lineas[i] = sangria + solucion
                    if i + 1 < len(lineas) and lineas[i + 1].strip() == "...":
                        del lineas[i + 1]
                    hechas += 1
        celda.source = "\n".join(lineas)
    return hechas


def main() -> int:
    nombres = sys.argv[1:] or sorted(p.name for p in NOTEBOOKS.glob("*.ipynb"))
    OUT.mkdir(parents=True, exist_ok=True)
    fallos = 0
    for nombre in nombres:
        nb = nbformat.read(NOTEBOOKS / nombre, as_version=4)
        if nombre.startswith("04") and rellenar(nb) == 0:
            print(f"{nombre}: aviso, no se ha encontrado el TODO del ejercicio 5 (¿ha cambiado el texto?)")
        inicio = time.time()
        try:
            NotebookClient(
                nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS)}}
            ).execute()
            estado = "OK"
        except Exception as exc:  # noqa: BLE001 — se informa y se sigue con el resto
            estado, fallos = f"ERROR\n{str(exc)[-1500:]}", fallos + 1
        nbformat.write(nb, OUT / nombre)
        print(f"{nombre}: {estado} ({time.time() - inicio:.0f} s)", flush=True)
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
