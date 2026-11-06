# Proceso de despliegue

## Chapuzón

Todos los servicios se despliegan con Chapuzón, nuestra herramienta interna de entrega continua. Cada merge a la rama main genera una imagen y Chapuzón la promueve primero a staging y después a producción con un despliegue canario del 10 % durante 15 minutos.

## Ventanas de despliegue

Se puede desplegar a producción de lunes a jueves de 09:00 a 18:00 (hora peninsular). Los viernes hay congelación de despliegues a partir de las 15:00 y durante todo el fin de semana. Los despliegues fuera de ventana necesitan la aprobación de la persona de guardia.

## Marcha atrás (rollback)

Para volver a una versión anterior se usa el comando `chapuzon rollback --to <sha> --service <nombre>`. El rollback no necesita aprobación y siempre está permitido, también fuera de ventana. Después de un rollback hay que abrir una incidencia en el canal #incidencias.

## Migraciones de base de datos

Las migraciones se despliegan por separado del código y deben ser compatibles hacia atrás durante al menos una versión (patrón expand/contract). Nunca se borra una columna en el mismo despliegue en el que se deja de usar.
