import cv2
import os
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from skimage.morphology import skeletonize
from skimage.util import invert
import math
import csv
#from scipy.interpolate import splprep, splev
import shutil
import re


PX_A_MM = 0.2127
PX_A_CM = 0.02127

def crear_directorio_si_no_existe(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def filtro_binaria(img, area_min=5000, area_max=120000, elong_umbral=2.2):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        for c in contornos:
            area = cv2.contourArea(c)
            if area_min < area < area_max:
                x, y, wc, hc = cv2.boundingRect(c)
                elong = max(hc / wc, wc / hc)
                elong_requerido = elong_umbral + (10000 / (area + 1e-5)) * 0.02
                if elong > elong_requerido and y < h * 0.7:
                    return True
    return False

def detectar_radicula(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    skel = skeletonize(invert(binary > 0)).astype(np.uint8)

    G = nx.Graph()
    coords = np.argwhere(skel == 1)
    # 🔹 Evitar grafos gigantes que congelan el proceso
    MAX_NODOS = 10000
    if len(coords) > MAX_NODOS:
        print(f"⚠️ Esqueleto demasiado grande ({len(coords)} nodos). Se omite detección automática.")
        return 0, img
    
    for (y, x) in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx or dy:
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1]:
                        if skel[ny, nx_] == 1:
                            G.add_edge((x, y), (nx_, ny))

    extremos = [n for n in G.nodes if G.degree(n) == 1]
    max_len, mejor_camino = 0, []
    
    # 🔹 Si hay demasiados extremos, abortar detección automática (para evitar cuelgues)
    MAX_EXTREMOS = 20
    if len(extremos) > MAX_EXTREMOS:
        print(f"⚠️ Demasiados extremos ({len(extremos)}). Se omite detección automática para evitar bloqueo.")
        return 0, img

    for i in range(len(extremos)):
        for j in range(i + 1, len(extremos)):
            try:
                path = nx.shortest_path(G, source=extremos[i], target=extremos[j])
                if len(path) > max_len:
                    max_len, mejor_camino = len(path), path
            except:
                continue

    img_out = img.copy()
    if len(mejor_camino) > 10:
        for k in range(len(mejor_camino) - 1):
            pt1, pt2 = mejor_camino[k], mejor_camino[k + 1]
            cv2.line(img_out, pt1, pt2, (255, 0, 0), 1)
        cv2.circle(img_out, mejor_camino[0], 2, (0, 0, 255), -1)
        cv2.circle(img_out, mejor_camino[-1], 2, (0, 255, 0), -1)
    else:
        max_len = 0

    return max_len, img_out

def analizar_radicula(BASE_DIR, FECHA=None, solo_faltantes=True, solo_carpetas=None):
    if FECHA is None:
        FECHA = datetime.now().strftime("%d%m%Y")

    PROC_DIR = os.path.join(BASE_DIR, "data/germinacion/data/procesadas/recortadas")
    RES_BASE = os.path.join(BASE_DIR, "data/germinacion/data/resultados")
    crear_directorio_si_no_existe(RES_BASE)

    # si me pasás una lista, la convertimos a set para filtrar rápido
    solo_set = set(solo_carpetas) if solo_carpetas else None

    for carpeta in sorted(os.listdir(PROC_DIR)):
        if not carpeta.startswith("recortes_"):
            continue

        # 👉 PROCESAR SOLO LAS CARPETAS INDICADAS (si vienen)
        if solo_set is not None and carpeta not in solo_set:
            continue

        carpeta_path = os.path.join(PROC_DIR, carpeta)
        RES_DIR = os.path.join(RES_BASE, carpeta)
        crear_directorio_si_no_existe(RES_DIR)

        celda_files = [f for f in sorted(os.listdir(carpeta_path)) if f.lower().endswith(".jpg")]
        csv_path = os.path.join(RES_DIR, f"{carpeta}_germinacion.csv")

        # evitar reprocesar celdas ya presentes
        df_old = None
        ya_procesadas = set()
        if os.path.exists(csv_path):
            try:
                df_old = pd.read_csv(csv_path)
                ya_procesadas = set(df_old["Celda"].astype(str).tolist())
            except Exception:
                pass

        # si está completo y me pediste solo faltantes, saltamos
        if solo_faltantes:
            res_files = [f for f in os.listdir(RES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")) and f.startswith("res_")]
            if os.path.exists(csv_path) and len(res_files) >= len(celda_files):
                print(f"⏭️ Saltando {carpeta}: ya procesada por completo.")
                continue

        resultados = []

        for fname in celda_files:
            if solo_faltantes and fname in ya_procesadas and os.path.exists(os.path.join(RES_DIR, f"res_{fname}")):
                continue

            img_path = os.path.join(carpeta_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            es_germinada = filtro_binaria(img)
            if es_germinada:
                long_px, img_out = detectar_radicula(img)
                estado = "GERMINADA" if long_px > 10 else "NO GERMINADA"
            else:
                long_px, img_out, estado = 0, img.copy(), "NO GERMINADA"

            long_mm = long_px * PX_A_MM
            long_cm = long_px * PX_A_CM

            long_r_cm_ini = round(long_cm, 2) if long_px > 0 and estado == "GERMINADA" else 0
            resultados.append([
                fname,
                long_px if long_px > 0 else "N/G",
                round(long_mm, 2) if long_px > 0 else "N/G",
                round(long_cm, 2) if long_px > 0 else "N/G",
                long_r_cm_ini,  # Long_r
                0,  # Long_h
                estado
            ])

            out_path = os.path.join(RES_DIR, f"res_{fname}")
            cv2.imwrite(out_path, img_out)

        if resultados:
            cols = ["Celda", "Longitud_px", "Longitud_mm", "Longitud_cm", "Long_r", "Long_h", "Estado"]
            df_new = pd.DataFrame(resultados, columns=cols)

            if df_old is not None:
                df_all = pd.concat([df_old, df_new], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=["Celda"], keep="last")
            else:
                df_all = df_new

            df_all.to_csv(csv_path, index=False)
            print(f"✅ CSV actualizado en {csv_path}")
        else:
            print(f"ℹ️ {carpeta}: no había celdas nuevas para procesar.")

"""
def analizar_radicula(BASE_DIR, FECHA=None):
    if FECHA is None:
        FECHA = datetime.now().strftime("%d%m%Y")

    PROC_DIR = os.path.join(BASE_DIR, "data/germinacion/data/procesadas/recortadas")
    RES_BASE = os.path.join(BASE_DIR, "data/germinacion/data/resultados")
    crear_directorio_si_no_existe(RES_BASE)

    for carpeta in sorted(os.listdir(PROC_DIR)):
        if not carpeta.startswith("recortes_"):
            continue

        carpeta_path = os.path.join(PROC_DIR, carpeta)
        RES_DIR = os.path.join(RES_BASE, carpeta)
        crear_directorio_si_no_existe(RES_DIR)

        resultados = []

        for fname in sorted(os.listdir(carpeta_path)):
            if not fname.endswith(".jpg"):
                continue

            img_path = os.path.join(carpeta_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            es_germinada = filtro_binaria(img)
            if es_germinada:
                long_px, img_out = detectar_radicula(img)
                estado = "GERMINADA" if long_px > 10 else "NO GERMINADA"
            else:
                long_px, img_out, estado = 0, img.copy(), "NO GERMINADA"

            long_mm = long_px * PX_A_MM
            long_cm = long_px * PX_A_CM

            resultados.append([
                fname,
                long_px if long_px > 0 else "N/G",
                round(long_mm, 2) if long_px > 0 else "N/G",
                round(long_cm, 2) if long_px > 0 else "N/G",
                0,  # Long_r (radícula)
                0,  # Long_h (hipocótilo)
                estado
            ])

            out_path = os.path.join(RES_DIR, f"res_{fname}")
            cv2.imwrite(out_path, img_out)

        # --- Generar CSV ---
        if resultados:
            import pandas as pd
            df = pd.DataFrame(resultados, columns=[
                "Celda",
                "Longitud_px",
                "Longitud_mm",
                "Longitud_cm",
                "Long_r",
                "Long_h",
                "Estado"
            ])
            csv_path = os.path.join(RES_DIR, f"{carpeta}_germinacion.csv")
            df.to_csv(csv_path, index=False)
            print(f"✅ CSV guardado en {csv_path}")
        else:
            print(f"⚠️ No se generaron resultados en {carpeta}")
"""

def detectar_ultimo_punto_azul(img):
    """
    Devuelve el punto más extremo del trazo azul detectado en la imagen calibrada.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 120, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Encontrar todos los píxeles del trazo azul
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    # Calcular extremos (por ejemplo el más abajo o más a la derecha)
    ultimo_punto = tuple(coords[coords[:, :, 1].argmax()][0])  # más abajo
    return ultimo_punto

def calibrar_r_simple(path_resultado):
    """
    Reabre una imagen calibrada y permite:
      - [C] Marcar codo: click en cualquier parte del trazo; se proyecta al polilínea.
      - Clicks: extender manualmente (azul).
      - [R] Reemplazo total: borra trazo previo; ENTER reemplaza total (ENTER sin puntos = N/G).
      - [ENTER] Guardar (suma o reemplaza según modo). Si hay codo, se particiona Long_r/Long_h
        garantizando Long_r + Long_h == Longitud_cm.
      - [ESC]/[Q] Salir sin guardar.
    """
    # -------- Helpers geométricos y de color (internos) --------
    def _blue_mask(img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 120, 50])
        upper_blue = np.array([130, 255, 255])
        return cv2.inRange(hsv, lower_blue, upper_blue)

    def _ordered_blue_path(img_bgr):
        """Polilínea del trazo azul como lista ORDENADA de (x,y)."""
        mask = _blue_mask(img_bgr)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return []

        pts = list(zip(xs.tolist(), ys.tolist()))  # (x,y)
        pts_set = set(pts)

        # grafo 8-conexo
        neigh = {p: [] for p in pts}
        for (x, y) in pts:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    q = (x + dx, y + dy)
                    if q in pts_set:
                        neigh[(x, y)].append(q)

        endpoints = [p for p, ns in neigh.items() if len(ns) == 1]
        start = endpoints[0] if endpoints else pts[0]

        # orientar por punto rojo si existe (inicio)
        red_mask = cv2.inRange(img_bgr, (0, 0, 200), (20, 20, 255))
        reds = cv2.findNonZero(red_mask)
        if reds is not None and len(endpoints) >= 1:
            rc = tuple(np.mean(reds.reshape(-1, 2), axis=0).astype(int))
            start = min(endpoints, key=lambda e: (e[0]-rc[0])**2 + (e[1]-rc[1])**2)

        ordered = [start]
        prev = None
        cur = start
        visited = {start}
        while True:
            nxts = [n for n in neigh[cur] if n != prev]
            n_candidate = None
            for n in nxts:
                if n not in visited:
                    n_candidate = n
                    break
            if n_candidate is None:
                break
            ordered.append(n_candidate)
            visited.add(n_candidate)
            prev, cur = cur, n_candidate
            if len(ordered) > 50000:  # safety
                break
        return ordered

    def _cumlen(pts):
        """Longitudes acumuladas (px) en cada vértice; len == len(pts)."""
        acc = [0.0]
        for i in range(1, len(pts)):
            acc.append(acc[-1] + math.dist(pts[i-1], pts[i]))
        return acc

    def _project_on_polyline(pts, p):
        """
        Proyecta punto p=(x,y) sobre el polilínea pts.
        Devuelve: (long_px_desde_inicio, punto_proyectado(x,y), seg_idx, t)
          seg_idx = índice del segmento [i, i+1] elegido
          t       = fracción en [0,1] dentro de ese segmento
        """
        if len(pts) < 2:
            return 0.0, (int(p[0]), int(p[1])), None, 0.0

        px, py = p
        best = (1e18, 0.0, (int(pts[0][0]), int(pts[0][1])), 0, 0.0)  # (dist2, len_px, proj, idx, t)

        # precomputo de longitudes acumuladas
        acc = _cumlen(pts)

        for i in range(len(pts)-1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            vx, vy = (x2 - x1), (y2 - y1)
            seg2 = vx*vx + vy*vy
            if seg2 == 0:
                # segmento degenerado
                qx, qy = x1, y1
                t = 0.0
            else:
                t = ((px - x1)*vx + (py - y1)*vy) / seg2
                t = max(0.0, min(1.0, t))
                qx = x1 + t * vx
                qy = y1 + t * vy

            d2 = (px - qx)**2 + (py - qy)**2
            len_px = acc[i] + math.dist((x1, y1), (qx, qy))
            if d2 < best[0]:
                best = (d2, len_px, (int(round(qx)), int(round(qy))), i, t)

        return best[1], best[2], best[3], best[4]

    def _len_list(pts):
        if len(pts) < 2:
            return 0.0
        return float(sum(math.dist(pts[i-1], pts[i]) for i in range(1, len(pts))))

    def _draw_cross(img, pt, size=6, color=(0, 255, 255), thickness=2):
        x, y = int(pt[0]), int(pt[1])
        cv2.line(img, (x-size, y), (x+size, y), color, thickness)
        cv2.line(img, (x, y-size), (x, y+size), color, thickness)

    # -------- Carga de imágenes y paths --------
    img_calibrada = cv2.imread(path_resultado)
    if img_calibrada is None:
        print(f"❌ No se pudo abrir la imagen calibrada:\n{path_resultado}")
        return None

    nombre_archivo = os.path.basename(path_resultado)
    match = re.search(r"celda_\d+", nombre_archivo)
    if not match:
        print(f"⚠️ No se pudo identificar el número de celda en: {nombre_archivo}")
        return None
    nombre_celda = match.group(0) + ".jpg"

    carpeta_resultados = os.path.dirname(path_resultado)
    carpeta_recortes = carpeta_resultados.replace(
        os.path.join("data", "germinacion", "data", "resultados"),
        os.path.join("data", "germinacion", "data", "procesadas", "recortadas")
    )
    ruta_original = os.path.join(carpeta_recortes, nombre_celda)
    if not os.path.exists(ruta_original):
        print(f"⚠️ No se encontró la copia sin calibrar:\n{ruta_original}")
        return None

    img_original = cv2.imread(ruta_original)
    img_display = img_calibrada.copy()

    # -------- Estado de edición --------
    base_path_pts = _ordered_blue_path(img_calibrada)   # polilínea existente
    path_pts = base_path_pts.copy()                     # base + extensión (o solo manual si R)
    puntos = []                                         # extensión manual (lista de (x,y))
    last_blue = None
    if len(base_path_pts) > 0:
        last_blue = base_path_pts[-1]
    replace_total = False

    codo_select_mode = False
    codo_len_px = None          # longitud proyectada (px) desde el inicio
    codo_point = None           # punto proyectado para dibujar la cruz

    print("Teclas: [R] Reemplazo total  [Z] Borrar último punto  [C] Marcar codo  [ENTER] Guardar  [ESC] Salir")
    # Ventana 1:1 (sin escalado) -> click == píxel
    #cv2.namedWindow("Calibración Manual", cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("Calibración Manual", img_display)~
    cv2.namedWindow("Calibracion Manual", cv2.WINDOW_NORMAL)
    h, w = img_display.shape[:2]
    target_h = 800  # alto inicial cómodo (podés cambiarlo)
    scale = max(1.0, min(3.0, target_h / max(1, h)))
    cv2.resizeWindow("Calibracion Manual", int(w * scale), int(h * scale))
    cv2.imshow("Calibracion Manual", img_display)

    # -------- Render --------
    def redibujar(base):
        view = base.copy()
        # extensión manual
        for i in range(1, len(puntos)):
            cv2.line(view, puntos[i-1], puntos[i], (255, 0, 0), 1)
        for p in puntos:
            cv2.circle(view, p, 1, (255, 0, 0), -1)

        if codo_point is not None:
            _draw_cross(view, codo_point)

        # HUD extensión
        if len(puntos) > 1:
            ext_cm = _len_list(puntos) * PX_A_CM
            overlay = view.copy()
            cv2.rectangle(overlay, (5, 5), (260, 35), (0, 0, 0), -1)
            view = cv2.addWeighted(overlay, 0.5, view, 0.5, 0)
            cv2.putText(view, f"Ext: {ext_cm:.2f} cm", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Calibracion Manual", view)

    # -------- Interacción --------
    def on_mouse(event, x, y, flags, param):
        nonlocal puntos, path_pts, codo_select_mode, codo_len_px, codo_point, replace_total, last_blue
        if event == cv2.EVENT_LBUTTONDOWN:
            if codo_select_mode:
                if len(path_pts) >= 2:
                    codo_len_px, codo_point, _, _ = _project_on_polyline(path_pts, (x, y))
                    print(f"✚ Codo proyectado: {codo_point}  |  len_r_px={codo_len_px:.2f}")
                else:
                    print("⚠ No hay trazo para proyectar el codo.")
                codo_select_mode = False
                base = img_original if replace_total else img_calibrada
                redibujar(base)
                return

            # extensión normal
            newp = (int(x), int(y))
            if not puntos and not replace_total and last_blue is not None:
                # enganchar al final del trazo existente
                puntos.append(last_blue)
                path_pts.append(last_blue)
            puntos.append(newp)
            path_pts.append(newp)
            last_blue = newp

            base = img_original if replace_total else img_calibrada
            redibujar(base)

    cv2.setMouseCallback("Calibracion Manual", on_mouse)

    csv_path = None
    while True:
        key = cv2.waitKey(0) & 0xFF
        if cv2.getWindowProperty("Calibracion Manual", cv2.WND_PROP_VISIBLE) < 1:
            print("🚪 Ventana cerrada — cancelado.")
            break

        if key in (ord('c'), ord('C')):
            if len(path_pts) >= 2:
                codo_select_mode = True
                print("🟡 Modo codo: hacé click sobre el trazo (se proyecta sobre la polilínea).")
            else:
                print("⚠ No hay trazo para marcar codo.")

        elif key in (ord('z'), ord('Z')):
            if puntos:
                last = puntos.pop()
                if path_pts and path_pts[-1] == last:
                    path_pts.pop()
                # si el codo cae fuera por borrar puntos, lo invalidamos
                if codo_point is not None and len(path_pts) < 2:
                    codo_point, codo_len_px = None, None
                base = img_original if replace_total else img_calibrada
                redibujar(base)
                print("↩ Último punto eliminado.")
            else:
                print("⚠ No hay puntos para borrar.")

        elif key in (ord('r'), ord('R')):
            # Reemplazo total
            replace_total = True
            puntos.clear()
            path_pts = []              # empezamos de cero
            codo_point = None
            codo_len_px = None
            last_blue = None
            base = img_original
            cv2.imshow("Calibracion Manual", base)
            print("🔄 REEMPLAZO TOTAL: dibujá desde cero. ENTER = reemplazar. ENTER sin puntos = N/G.")

        elif key == 13:  # ENTER
            # elegir base de guardado
            base = img_original if replace_total else img_calibrada
            out = base.copy()

            # dibujar extensión y codo (si hay)
            for i in range(1, len(puntos)):
                cv2.line(out, puntos[i-1], puntos[i], (255, 0, 0), 1)
            for p in puntos:
                cv2.circle(out, p, 1, (255, 0, 0), -1)
            if codo_point is not None:
                _draw_cross(out, codo_point)
            cv2.imwrite(path_resultado, out)
            print(f"✅ Imagen guardada: {path_resultado}")

            # actualizar CSV
            csv_files = [f for f in os.listdir(carpeta_resultados) if f.lower().endswith(".csv")]
            if not csv_files:
                print("⚠ No encontré CSV en la carpeta de resultados.")
                break
            csv_path = os.path.join(carpeta_resultados, csv_files[0])

            try:
                df = pd.read_csv(csv_path)
                mask = df["Celda"].astype(str).str.lower() == nombre_celda.lower()

                # asegurar numéricas
                for col in ["Longitud_px", "Longitud_mm", "Longitud_cm"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                if replace_total:
                    # --- REEMPLAZO TOTAL ---
                    if len(puntos) > 1:
                        total_px = _len_list(puntos)
                        total_cm = round(total_px * PX_A_CM, 2)
                        df.loc[mask, "Longitud_px"] = round(total_px, 2)
                        df.loc[mask, "Longitud_mm"] = round(total_px * PX_A_MM, 2)
                        df.loc[mask, "Longitud_cm"] = total_cm
                        df.loc[mask, "Estado"] = "GERMINADA"
                    else:
                        # N/G
                        df.loc[mask, ["Longitud_px", "Longitud_mm", "Longitud_cm"]] = "N/G"
                        df.loc[mask, "Estado"] = "NO GERMINADA"

                else:
                    # --- EXTENSIÓN ---
                    ext_px = _len_list(puntos)
                    prev_px = df.loc[mask, "Longitud_px"].fillna(0)
                    df.loc[mask, "Longitud_px"] = round(prev_px + ext_px, 2)
                    df.loc[mask, "Longitud_mm"] = round(df.loc[mask, "Longitud_px"] * PX_A_MM, 2)
                    df.loc[mask, "Longitud_cm"] = round(df.loc[mask, "Longitud_px"] * PX_A_CM, 2)
                    df.loc[mask, "Estado"] = "GERMINADA"

                # columnas R/H
                if "Long_r" not in df.columns:
                    df["Long_r"] = 0.0
                if "Long_h" not in df.columns:
                    df["Long_h"] = 0.0
                df["Long_r"] = pd.to_numeric(df["Long_r"], errors="coerce").fillna(0.0)
                df["Long_h"] = pd.to_numeric(df["Long_h"], errors="coerce").fillna(0.0)

                # asignar R/H según haya codo o no
                total_cm_val = df.loc[mask, "Longitud_cm"].values[0]
                if isinstance(total_cm_val, str):
                    # N/G en reemplazo total → R/H = 0
                    df.loc[mask, ["Long_r", "Long_h"]] = 0.0
                else:
                    total_cm = float(total_cm_val)
                    if codo_len_px is not None and len(path_pts) >= 2:
                        # con codo → se particiona R/H
                        r_cm = round(codo_len_px * PX_A_CM, 2)
                        r_cm = max(0.0, min(r_cm, total_cm))
                        h_cm = round(total_cm - r_cm, 2)
                    else:
                        # sin codo → R = total, H = 0 (lo que pediste)
                        r_cm, h_cm = total_cm, 0.0
                    df.loc[mask, "Long_r"] = r_cm
                    df.loc[mask, "Long_h"] = h_cm

                df.to_csv(csv_path, index=False)
                print("📄 CSV actualizado.")
            except Exception as e:
                print(f"❌ Error al actualizar CSV: {e}")
            break

        elif key in (27, ord('q')):
            print("🚪 Cancelado sin guardar.")
            break

    cv2.destroyAllWindows()
    return csv_path
