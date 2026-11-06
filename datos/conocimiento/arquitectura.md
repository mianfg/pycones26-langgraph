# Arquitectura de servicios

## Visión general

La plataforma está formada por servicios en Python 3.15 desplegados en Kubernetes. Se comunican mediante HTTP interno y una cola de eventos en Redis Streams. La base de datos principal es PostgreSQL 18.

## Servicios

Hucha gestiona pagos y reembolsos. Periscopio es el buscador del catálogo y el índice de stock. Graznido envía notificaciones (correo, push y SMS) y gestiona las alertas de guardia. Flotador gestiona el almacén: pedidos, etiquetas de envío y devoluciones. Nido es el panel web para los artistas que diseñan patitos.

## Equipos responsables

El equipo de Pagos es responsable de Hucha. El equipo de Catálogo lleva Periscopio. El equipo de Plataforma mantiene Graznido, Chapuzón y la infraestructura. El equipo de Almacén lleva Flotador y la tienda Estanque.
