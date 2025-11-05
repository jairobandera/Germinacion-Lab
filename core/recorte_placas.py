import cv2
import numpy as np
import os
import shutil
from datetime import datetime

def recortar_placas(BASE_DIR, FECHA=None):
    if FECHA is None:
        FECHA = datetime.now().strftime("%d%m%Y")

    BASE_GERMINACION = os.path.join(BASE_DIR, "data/germinacion/data")
    ORIG_BASE = os.path.join(BASE_GERMINACION, "originales")
    PROC_DIR = os.path.join(BASE_GERMINACION, "procesadas/placa_recortada")
    os.makedirs(ORIG_BASE, exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)

    DATA_DIR = os.path.join(ORIG_BASE, f"originales_{FECHA}")
    os.makedirs(DATA_DIR, exist_ok=True)

    for file in os.listdir(ORIG_BASE):
        full_path = os.path.join(ORIG_BASE, file)
        if file.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(full_path):
            shutil.move(full_path, os.path.join(DATA_DIR, file))

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(DATA_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        edges = cv2.Canny(blur, 40, 120)
        kernel = np.ones((7,7), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_rect = None
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            area = w_box * h_box
            ratio = w_box / (h_box + 1e-5)
            if area > max_area and 0.8 < ratio < 2.2 and area > (h*w)*0.3:
                max_area = area
                best_rect = (x, y, w_box, h_box)

        if best_rect is None:
            continue

        top_ratio, bottom_ratio, left_ratio, right_ratio = 0.06, 0.38, 0.05, 0.04
        y1, y2 = int(h * top_ratio), int(h * (1 - bottom_ratio))
        x1, x2 = int(w * left_ratio), int(w * (1 - right_ratio))
        crop = img[y1:y2, x1:x2]

        nombre_salida = f"placa_recortada_{os.path.splitext(filename)[0]}_{FECHA}.jpg"
        out_path = os.path.join(PROC_DIR, nombre_salida)
        cv2.imwrite(out_path, crop)
