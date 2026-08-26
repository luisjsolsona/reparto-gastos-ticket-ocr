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

## Backend (opcional): OCR con PaddleOCR

Se probó primero [Donut](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2)
(modelo end-to-end que devuelve JSON estructurado), pero en tickets españoles
con varias columnas de precio mezclaba filas e inventaba cantidades y
precios — peor que un OCR clásico. Se sustituyó por
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), que solo hace lo que
tiene que hacer (detectar y leer texto) con bastante más precisión que
Tesseract.js, especialmente en fotos giradas, con brillos o en ángulo.

`backend/app.py` expone un endpoint que:

1. Detecta cada fragmento de texto de la imagen con PaddleOCR
2. Agrupa los fragmentos que están a la misma altura (misma fila de la
   tabla del ticket) y los ordena de izquierda a derecha, reconstruyendo
   líneas como `2 * VERDEJO 1.80€ 3.60€`
3. Devuelve ese texto reconstruido, listo para pasarlo por las mismas
   reglas de interpretación (cantidad · producto · precio) que ya usa el
   frontend con Tesseract.js

### Requisitos

- Docker y Docker Compose
- CPU es suficiente para uso personal (PaddleOCR es más ligero que un
  modelo de visión tipo Donut)

### Levantarlo

```bash
cd backend
docker compose up --build -d
docker compose logs -f
```

La primera vez descarga los modelos de detección/reconocimiento/orientación
de PaddleOCR (unos pocos MB, mucho menos que Donut). Cuando en los logs
aparezca `Uvicorn running on http://0.0.0.0:8000`, el servicio está listo en
el puerto `8010` del host.

### Probarlo

```bash
curl -X POST -F "file=@/ruta/a/tu/ticket.jpg" http://localhost:8010/ocr
```

Devuelve:

```json
{
  "text": "3 Agua Fuente Liviana 10,50 €\n1 Free Damm Limón 1/3 3,20 €\n...",
  "raw_lines": ["3 Agua Fuente Liviana 10,50 €", "..."]
}
```

El campo `text` es el que se pasaría a la misma lógica de parseo
(`preprocessLines` / `parseLineToItem`) que ya usa `frontend/Repartimos-gastos-ticket.html`.

### Estado

Validado con dos tickets reales de restaurante: mejora notable respecto a
Tesseract.js, especialmente en tickets con columnas de precio (Precio
unidad + Total línea), aunque el ajuste fino de resolución/detección puede
seguir necesitando algún retoque según el ticket.

### Conectar el frontend

`frontend/Repartimos-gastos-ticket.html` tiene un campo "Backend OCR
(opcional)" donde se pega la URL del endpoint (por ejemplo
`http://192.168.0.100:8010/ocr` en tu red local). Si el campo está relleno,
el escaneo intenta primero ese servidor; si no responde (apagado, fuera de
la red local, CORS bloqueado, etc.), cae automáticamente a Tesseract.js en
el navegador sin que el usuario tenga que hacer nada. Si se deja vacío,
siempre usa Tesseract.js.

## Autor

[Luis Solsona](https://github.com/luisjsolsona) · [mistikedu.com](https://mistikedu.com)
