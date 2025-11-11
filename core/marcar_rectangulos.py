import cv2
import numpy as np
import os
from datetime import datetime

# Extensiones de imagen válidas
VALID_EXTS = (".jpg", ".jpeg", ".png", ".jfif", ".bmp", ".tif", ".tiff")


def marcar_rectangulos(BASE_DIR, FECHA=None):
    if FECHA is None:
        FECHA = datetime.now().strftime("%d%m%Y")

    PROC_BASE = os.path.join(BASE_DIR, "data/germinacion/data/procesadas/placa_recortada")

    if not os.path.exists(PROC_BASE):
        return

    rows, cols = 2, 10
    cell_width, cell_height = 135, 340
    top_offset, left_offset = 45, -5
    horizontal_gap, vertical_gap = 51, 85
    rotation_angle = 1.2

    def draw_rotated_rectangle(image, x, y, w, h, angle_deg):
        rect_center = (x + w//2, y + h//2)
        rect = ((rect_center[0], rect_center[1]), (w, h), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(image, [box], 0, (0,255,0), 2)
        return box

    for filename in sorted(os.listdir(PROC_BASE)):
        if not filename.lower().endswith(VALID_EXTS):
            continue
        img_path = os.path.join(PROC_BASE, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        vis = img.copy()

        cell_num = 1
        for r in range(rows):
            for c in range(cols):
                x1 = left_offset + c * (cell_width + horizontal_gap)
                y1 = top_offset + r * (cell_height + vertical_gap)
                draw_rotated_rectangle(vis, x1, y1, cell_width, cell_height, rotation_angle)
                cv2.putText(vis, str(cell_num), (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cell_num += 1
