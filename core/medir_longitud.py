import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
IMG_PATH = "0% - 2.jpg"  # nombre de tu imagen en la misma carpeta

# === CARGAR IMAGEN ===
img = cv2.imread(IMG_PATH)
if img is None:
    print("❌ No se pudo abrir la imagen. Verificá el nombre o la ruta.")
    exit()

# === MOSTRAR Y ESPERAR CLICS ===
plt.figure(figsize=(10,5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Haz clic en los extremos de un segmento de la regla (por ejemplo, 0 cm y 10 cm)")
pts = plt.ginput(2, timeout=0)  # espera hasta que hagas 2 clics
plt.close()

if len(pts) != 2:
    print("❌ No se detectaron dos clics. Volvé a ejecutar y marca los dos puntos.")
    exit()

# === CALCULAR DISTANCIA ===
(x1, y1), (x2, y2) = pts
dist_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

dist_mm = 100  # distancia real entre puntos (10 cm = 100 mm)
px_to_mm = dist_mm / dist_px
px_to_cm = px_to_mm / 10

print("\n✅ RESULTADOS:")
print(f"Distancia seleccionada: {dist_px:.2f} píxeles")
print(f"1 px = {px_to_mm:.4f} mm = {px_to_cm:.5f} cm")
