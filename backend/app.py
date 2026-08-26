import io
import re
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from bs4 import BeautifulSoup
import numpy as np
from paddleocr import PaddleOCR, PPStructure

app = FastAPI(title="Ticket OCR (PaddleOCR)")

# Permite que el frontend (servido desde otro origen: GitHub Pages, mistikedu.com,
# o abierto como archivo local) pueda llamar a este backend desde el navegador.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "ok", "note": "frontend no incluido en esta imagen"}



# 'es' cubre el alfabeto latino (español incluido) en los modelos de PaddleOCR.
# det_limit_side_len sube el límite de resolución interno del detector: por
# defecto PaddleOCR reescala la imagen a ~960px de lado antes de detectar el
# texto, lo que en fotos de móvil de alta resolución aplasta tanto las líneas
# de un ticket térmico que dos renglones vecinos acaban tocándose y el
# detector los funde en un solo bloque. Con un límite más alto conserva
# resolución suficiente para separarlos.
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="es",
    show_log=False,
    det_limit_side_len=2500,
    det_db_unclip_ratio=1.6,
)

# PP-StructureV3: además de leer el texto, intenta reconocer la ESTRUCTURA de
# la tabla (filas y columnas) en vez de que tengamos que reconstruirla a
# mano por posición. Cuando detecta una tabla, suele resolver justo el
# problema de alineación entre la columna del nombre y la del precio.
table_engine = PPStructure(table=True, ocr=True, show_log=False, lang="es")


def table_html_to_lines(html):
    """Convierte el HTML de tabla que devuelve PP-Structure en líneas de
    texto (una por fila), uniendo el contenido de las celdas en orden."""
    soup = BeautifulSoup(html, "html.parser")
    lines = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" ".join(cells))
    return lines


# Un fragmento que es SOLO un número con decimales (con o sin símbolo de
# moneda pegado, que a veces el OCR lee mal como @, & o £) se trata como
# "columna de precio", no como parte del nombre del producto.
PRICE_TOKEN_RE = re.compile(r'^\d{1,4}[.,]\d{1,2}\s*[€@&£]?$')


def group_into_lines(pairs, y_tol_ratio=0.5):
    """PaddleOCR detecta texto por cajas sueltas, no por filas, y en tickets
    reales la columna de números no siempre está perfectamente alineada en
    altura con la columna del nombre del producto (un pequeño desajuste
    vertical entre columnas). Por eso el texto se trata en dos pasos:

    1) Los fragmentos que NO son un precio (el nombre, "2 * VERDEJO", etc.)
       se agrupan en filas por altura, comparando solo contra la última fila
       creada, para reconstruir el orden de lectura de la tabla.
    2) Cada fragmento que SÍ es un precio se asigna a la fila de producto
       más cercana en altura (no a una banda fija), lo que tolera ese
       pequeño desajuste vertical entre columnas.
    """
    items = []
    for box, (text, conf) in pairs:
        raw_text = text.strip()
        if not raw_text:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({
            "text": raw_text,
            "y": sum(ys) / len(ys),
            "h": max(ys) - min(ys) or 1,
            "x": min(xs),
        })

    name_items = [it for it in items if not PRICE_TOKEN_RE.match(it["text"])]
    price_items = [it for it in items if PRICE_TOKEN_RE.match(it["text"])]
    name_items.sort(key=lambda i: i["y"])

    rows = []
    for it in name_items:
        if rows:
            last = rows[-1]
            last_y = sum(w["y"] for w in last) / len(last)
            last_h = sum(w["h"] for w in last) / len(last)
            if abs(it["y"] - last_y) < last_h * y_tol_ratio:
                last.append(it)
                continue
        rows.append([it])

    row_objs = [
        {"words": r, "prices": [], "y": sum(w["y"] for w in r) / len(r)}
        for r in rows
    ]

    for pit in price_items:
        if row_objs:
            best_row = min(row_objs, key=lambda r: abs(pit["y"] - r["y"]))
            best_row["prices"].append(pit)
        else:
            row_objs.append({"words": [], "prices": [pit], "y": pit["y"]})

    row_objs.sort(key=lambda r: r["y"])

    text_lines = []
    for row in row_objs:
        words_sorted = sorted(row["words"], key=lambda w: w["x"])
        prices_sorted = sorted(row["prices"], key=lambda w: w["x"])
        parts = [w["text"] for w in words_sorted] + [w["text"] for w in prices_sorted]
        if parts:
            text_lines.append(" ".join(parts))
    return text_lines


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image)

    # 1) intenta reconocer la tabla completa (filas/columnas) con PP-Structure
    table_lines = []
    try:
        structure_result = table_engine(arr)
        for region in structure_result:
            if region.get("type") == "table":
                html = (region.get("res") or {}).get("html", "")
                if html:
                    table_lines.extend(table_html_to_lines(html))
    except Exception:
        table_lines = []

    if table_lines:
        return {"text": "\n".join(table_lines), "raw_lines": table_lines, "engine": "table"}

    # 2) si no se detectó ninguna tabla, cae al OCR de texto suelto +
    # reconstrucción de filas por posición (el método anterior)
    result = ocr_engine.ocr(arr, cls=True)
    boxes_texts = result[0] if result and result[0] else []
    pairs = [(bt[0], bt[1]) for bt in boxes_texts]

    lines = group_into_lines(pairs)
    return {"text": "\n".join(lines), "raw_lines": lines, "engine": "text"}
