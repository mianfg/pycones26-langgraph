"""Los grafos que sirve `langgraph dev` (langgraph.json → taller/studio.py) cargan y tienen la forma esperada."""

import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_langgraph_json_apunta_a_grafos_existentes():
    import taller.studio as studio

    config = json.loads((ROOT / "langgraph.json").read_text(encoding="utf-8"))
    for graph_id, ruta in config["graphs"].items():
        fichero, variable = ruta.split(":")
        assert (ROOT / fichero).exists(), ruta
        grafo = getattr(studio, variable)
        assert "__start__" in grafo.get_graph().nodes, graph_id
    assert set(config["graphs"]) == {"agente_basico", "rag", "agente_que_aprende"}


def test_los_grafos_del_servidor_no_llevan_persistencia_propia():
    # El Agent Server pone su checkpointer y su store; los grafos se compilan sin ellos.
    import taller.studio as studio

    for nombre in ("agente_basico", "rag", "agente_que_aprende"):
        grafo = getattr(studio, nombre)
        assert grafo.checkpointer is None, nombre
        assert grafo.store is None, nombre


def test_embed_del_store_tiene_las_dimensiones_de_langgraph_json():
    import taller.studio as studio

    dims = json.loads((ROOT / "langgraph.json").read_text(encoding="utf-8"))["store"]["index"]["dims"]
    vectores = asyncio.run(studio.embed(["guardias", "despliegue"]))
    assert [len(v) for v in vectores] == [dims, dims] == [studio.STUDIO_DIMS] * 2
    assert abs(sum(x * x for x in vectores[0]) - 1.0) < 1e-6
