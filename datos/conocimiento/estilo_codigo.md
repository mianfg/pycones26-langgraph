# Guía de estilo de código

## Herramientas

Usamos uv para gestionar dependencias y entornos, ruff para formateo y lint, y pytest para los tests. Las anotaciones de tipos son obligatorias en el código nuevo y se comprueban en CI.

## Revisiones de código

Toda pull request necesita una aprobación de alguien de otro equipo si toca Hucha o Flotador. Las pull requests no deben superar 400 líneas cambiadas; si son más grandes, hay que dividirlas.

## Secretos

Nunca se suben secretos al repositorio. Los secretos se guardan en Bóveda, nuestro gestor de secretos, y se inyectan como variables de entorno en el despliegue. Si se filtra un secreto, hay que rotarlo inmediatamente desde Bóveda y abrir una incidencia SEV2.
