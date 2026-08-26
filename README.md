# Repartimos gastos (ticket OCR)

Herramienta para repartir la cuenta de un restaurante entre varios comensales,
producto a producto y unidad a unidad, a partir de una foto del ticket.

## Estructura

```
frontend/   → Repartimos-gastos-ticket.html (app de reparto, un solo archivo, sin dependencias de servidor)
backend/    → microservicio de OCR con Donut (opcional, mejora la lectura de tickets)
```

## Frontend

`frontend/Repartimos-gastos-ticket.html` es una app de una sola página (HTML/CSS/JS,
sin build ni dependencias de servidor) para:

- Escanear un ticket (con recorte manual antes de leerlo)
- Revisar y corregir los productos detectados antes de usarlos
- Añadir comensales y asignarles unidades de cada producto, una a una
- Repartir unidades sueltas "entre todos" (ej. una ronda de aguas)
- Comprobar que la suma coincide con el total impreso del ticket
- Editar o eliminar productos después de cargarlos, y reiniciar todo

Por defecto usa [Tesseract.js](https://github.com/naptha/tesseract.js) para el
OCR, ejecutado en el propio navegador (sin backend, sin subir la foto a
ningún sitio). Funciona razonablemente en tickets nítidos, pero tiene
limitaciones claras en papel térmico con brillos, ángulo o tablas con varias
columnas de precio.

Para usarlo: abre el `.html` directamente en el navegador (doble clic, o
súbelo a cualquier hosting estático).

## Backend (opcional): OCR estructurado con Donut

`backend/` monta un microservicio en Docker con
[Donut](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2)
(`naver-clova-ix/donut-base-finetuned-cord-v2`), un modelo gratuito de
Hugging Face entrenado específicamente en tickets de compra. A diferencia de
un OCR clásico + expresiones regulares, Donut devuelve directamente un JSON
estructurado con las líneas del ticket (nombre, cantidad, precio), sin
necesidad de parsear texto plano.

### Requisitos

- Docker y Docker Compose
- ~2 GB libres para el modelo y las dependencias (se descarga la primera vez)
- CPU es suficiente para uso personal; con GPU la respuesta es más rápida

### Levantarlo

```bash
cd backend
docker compose up --build -d
docker compose logs -f
```

La primera vez tardará en descargar el modelo (~800 MB). Cuando en los logs
aparezca `Uvicorn running on http://0.0.0.0:8000`, el servicio está listo en
el puerto `8010` del host.

### Probarlo

```bash
curl -X POST -F "file=@/ruta/a/tu/ticket.jpg" http://localhost:8010/ocr
```

Devuelve algo así (estructura del dataset original, en inglés/campos cortos):

```json
{
  "raw": {
    "menu": [
      {"nm": "AGUA FUENTE LIVIANA", "cnt": "3", "price": "10,50"},
      ...
    ],
    "total": {"total_price": "215,28"}
  }
}
```

### Estado

Este backend es **experimental**: el modelo está entrenado sobre tickets de
tiendas (dataset CORD), no específicamente sobre tickets de hostelería
española, así que su precisión real en este caso de uso todavía se está
validando. El frontend, de momento, sigue funcionando de forma autónoma con
Tesseract.js; la idea es que hable con este backend en cuanto se confirme que
mejora los resultados de forma consistente.

## Autor

[Luis Solsona](https://github.com/luisjsolsona) · [mistikedu.com](https://mistikedu.com)
