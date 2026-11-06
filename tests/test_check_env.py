"""`check_env` en modo offline: sin red, tiene que acabar en «Todo listo.»."""

from taller.config import check_env


def test_check_env_sin_modelos(capsys):
    assert check_env(call_models=False)
    salida = capsys.readouterr().out
    assert "OK     LLM_MODEL = offline:extractivo" in salida
    assert "Todo listo" in salida


def test_check_env_completo(capsys):
    assert check_env()
    salida = capsys.readouterr().out
    assert "ERROR" not in salida
    for linea in (
        "responde",
        "sabe pedir herramientas",
        "salida estructurada",
        "embeddings de 256",
        "34 fragmentos",
        "checkpointer SQLite",
        # Sin LANGSMITH_API_KEY: aviso, nunca error (cuenta y clave son opcionales).
        "AVISO  sin LANGSMITH_API_KEY: opcional",
    ):
        assert linea in salida, linea
    assert salida.rstrip().endswith("el taller.)") or salida.rstrip().endswith("Todo listo.")
