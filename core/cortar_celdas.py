import cv2
import numpy as np
import os
from datetime import datetime

# Extensiones de imagen válidas
VALID_EXTS = (".jpg", ".jpeg", ".png", ".jfif", ".bmp", ".tif", ".tiff")


def cortar_celdas(BASE_DIR, FECHA=None):
    if FECHA is None:
        FECHA = datetime.now().strftime("%d%m%Y")

    PROC_DIR = os.path.join(BASE_DIR, "data/germinacion/data/procesadas/placa_recortada")
    OUT_BASE = os.path.join(BASE_DIR, "data/germinacion/data/procesadas/recortadas")
    os.makedirs(OUT_BASE, exist_ok=True)

    for placa in os.listdir(PROC_DIR):
        if not placa.lower().endswith((VALID_EXTS)):
            continue

        nombre_base = os.path.splitext(placa)[0]
        OUT_DIR = os.path.join(OUT_BASE, f"recortes_{nombre_base}_{FECHA}")
        os.makedirs(OUT_DIR, exist_ok=True)

        img_path = os.path.join(PROC_DIR, placa)
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]

        # === parámetros ===
        rows, cols = 2, 10
        cell_width, cell_height = 135, 340
        top_offset, left_offset = 45, -5
        horizontal_gap, vertical_gap = 51, 85
        rotation_angle = 1.2

        def get_rotated_box(x, y, w, h, angle_deg):
            rect_center = (x + w // 2, y + h // 2)
            rect = ((rect_center[0], rect_center[1]), (w, h), angle_deg)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            return box

        def extract_rotated_region(img, box):
            rect = cv2.minAreaRect(box)
            center, size, angle = rect
            size = (int(size[0]), int(size[1]))
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            x, y = int(center[0] - size[0] / 2), int(center[1] - size[1] / 2)
            crop = rotated[y:y + size[1], x:x + size[0]]
            return crop

        celdas_guardadas = 0
        cell_num = 1

        for r in range(rows):
            for c in range(cols):
                x = left_offset + c * (cell_width + horizontal_gap)
                y = top_offset + r * (cell_height + vertical_gap)

                box = get_rotated_box(x, y, cell_width, cell_height, rotation_angle)
                crop = extract_rotated_region(img, box)

                # === nuevo sistema: clamp dentro de límites ===
                x0 = max(0, int(x))
                y0 = max(0, int(y))
                x1 = min(W, int(x + cell_width))
                y1 = min(H, int(y + cell_height))

                w = x1 - x0
                h = y1 - y0

                MIN_FRAC = 0.5
                if w < cell_width * MIN_FRAC or h < cell_height * MIN_FRAC:
                    print(f"⚠️ Celda mayormente fuera del área (fila {r+1}, col {c+1}) — omitida.")
                    continue

                crop = img[y0:y1, x0:x1]

                # --- recorte de bordes blancos ---
                h_crop, w_crop = crop.shape[:2]
                margin = 3
                if h_crop > margin * 2 and w_crop > margin * 2:
                    crop = crop[margin:h_crop - margin, margin:w_crop - margin]

                ruta_salida = os.path.join(OUT_DIR, f"celda_{cell_num:02d}.jpg")
                cv2.imwrite(ruta_salida, crop)
                celdas_guardadas += 1
                cell_num += 1

        print(f"✅ {nombre_base}: {celdas_guardadas}/{rows * cols} celdas guardadas.")
