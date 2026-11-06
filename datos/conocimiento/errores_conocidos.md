# Catálogo de errores conocidos

## ERR-4012: tiempo de espera en la pasarela de pago

Lo emite Hucha, el servicio de pagos, cuando la pasarela bancaria tarda más de 8 segundos en responder. Hucha reintenta automáticamente 2 veces con espera exponencial. Si la tasa de ERR-4012 supera el 5 % durante 10 minutos, es una SEV1 y hay que avisar a la guardia de Pagos.

## ERR-5003: stock desincronizado

Aparece cuando Periscopio, el buscador del catálogo, muestra como disponibles patitos que ya se han agotado; es típico en los lanzamientos de ediciones limitadas. Se arregla lanzando la tarea `reindex-producto <sku>` desde el panel de Nido. No requiere aviso a la guardia salvo que afecte a más de 50 referencias.

## ERR-7781: etiqueta de envío duplicada

Dos pedidos con la misma etiqueta de envío. Es grave porque bloquea la línea de empaquetado del almacén. Hay que anular ambas etiquetas, regenerarlas con `flotador-cli etiquetas regenerar` y avisar a Atención al Cliente.

## GRZ-210: límite de peticiones de Graznido

Graznido ha superado el límite de 500 notificaciones por segundo. Las notificaciones se encolan y se envían con retraso. Normalmente se resuelve solo en menos de 5 minutos.
