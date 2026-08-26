import io

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

app = FastAPI(title="Ticket OCR (PaddleOCR)")

# 'es' cubre el alfabeto latino (español incluido) en los modelos de PaddleOCR
ocr_engine = PaddleOCR(use_angle_cls=True, lang="es", show_log=False)


def group_into_lines(pairs, y_tol_ratio=0.6):
    """PaddleOCR detecta texto por cajas sueltas, no por filas. Esto agrupa
    las cajas que están a la misma altura (misma fila de la tabla del
    ticket) y las ordena de izquierda a derecha, para reconstruir una línea
    de texto como '2 * VERDEJO 1.80€ 3.60€' en vez de tres fragmentos
    sueltos."""
    items = []
    for box, (text, conf) in pairs:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({
            "text": text,
            "y": sum(ys) / len(ys),
            "h": max(ys) - min(ys) or 1,
            "x": min(xs),
        })

    items.sort(key=lambda i: i["y"])

    lines = []
    for it in items:
        placed = False
        for line in lines:
            avg_y = sum(w["y"] for w in line) / len(line)
            avg_h = sum(w["h"] for w in line) / len(line)
            if abs(it["y"] - avg_y) < avg_h * y_tol_ratio:
                line.append(it)
                placed = True
                break
        if not placed:
            lines.append([it])

    lines.sort(key=lambda line: sum(w["y"] for w in line) / len(line))
    text_lines = []
    for line in lines:
        line.sort(key=lambda w: w["x"])
        text_lines.append(" ".join(w["text"] for w in line))
    return text_lines


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image)

    result = ocr_engine.ocr(arr, cls=True)
    boxes_texts = result[0] if result and result[0] else []
    pairs = [(bt[0], bt[1]) for bt in boxes_texts]

    lines = group_into_lines(pairs)
    return {"text": "\n".join(lines), "raw_lines": lines}
