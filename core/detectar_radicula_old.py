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
                estado
            ])

            out_path = os.path.join(RES_DIR, f"res_{fname}")
            cv2.imwrite(out_path, img_out)

        if resultados:
            import pandas as pd
            df = pd.DataFrame(resultados, columns=["Celda", "Longitud_px", "Longitud_mm", "Longitud_cm", "Estado"])
            csv_path = os.path.join(RES_DIR, f"{carpeta}_germinacion.csv")
            df.to_csv(csv_path, index=False)

def calibrar_radicula_manual(carpeta_recortes):
    """
    Modo manual: usa la detección automática como BASE (verde)
    y solo SUMA la extensión que el usuario agregue (azul).
    - ENTER: confirma (guarda imagen y escribe CSV)
    - R: borra SOLO la extensión azul (conserva la base verde)
    - Z: deshacer último punto de la extensión
    - ESC: sale
    """

    if not os.path.exists(carpeta_recortes):
        print(f"❌ Carpeta no encontrada: {carpeta_recortes}")
        return

    imagenes = [f for f in os.listdir(carpeta_recortes)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not imagenes:
        print("❌ No se encontraron imágenes en la carpeta indicada.")
        return

    # === Crear carpeta de resultados igual que el automático ===
    # Buscar la raíz del proyecto a partir de "data/germinacion"
    if "data" in carpeta_recortes:
        base_dir = carpeta_recortes.split("data")[0]
    else:
        base_dir = os.path.dirname(carpeta_recortes)

    # Directorio base de resultados
    RES_BASE = os.path.join(base_dir, "data", "germinacion", "data", "resultados")

    # Crear si no existe
    os.makedirs(RES_BASE, exist_ok=True)

    # Nombre de carpeta (igual que en las otras funciones)
    nombre_carpeta = os.path.basename(carpeta_recortes.strip("/\\"))

    # Carpeta final de salida dentro de resultados
    out_dir = os.path.join(RES_BASE, nombre_carpeta)
    os.makedirs(out_dir, exist_ok=True)

    # Ruta del CSV de salida
    csv_path = os.path.join(out_dir, f"{nombre_carpeta}_manual.csv")

    print(f"📂 Guardando resultados manuales en: {out_dir}")

    print("🧠 Iniciando detección automática + calibración manual (suma solo extensión azul)")
    print("🔹 Click: extender línea (azul)")
    print("🔹 Z: deshacer último punto   🔹 R: borrar extensión   🔹 ENTER: confirmar   🔹 ESC: salir")

    resultados = []

    # ------------ Helpers ------------
    def _dist_total(pts):
        if len(pts) < 2:
            return 0.0
        return sum(math.dist(pts[i], pts[i+1]) for i in range(len(pts)-1))

    def _dibujar(img, auto_pts, ext_pts):
        # auto (VERDE)
        if auto_pts:
            for i in range(1, len(auto_pts)):
                cv2.line(img, auto_pts[i-1], auto_pts[i], (0, 255, 0), 2)
            cv2.circle(img, auto_pts[0], 3, (0, 0, 255), -1)   # inicio rojo
            cv2.circle(img, auto_pts[-1], 3, (0, 255, 0), -1)  # fin verde

        # extensión (AZUL), partiendo desde el último punto de auto si existe
        if ext_pts:
            inicio = auto_pts[-1] if auto_pts else ext_pts[0]
            prev = inicio
            for p in ext_pts:
                cv2.line(img, prev, p, (255, 0, 0), 2)
                prev = p

    # ------------ Interacción ------------
    cv2.namedWindow("Calibración Manual", cv2.WINDOW_NORMAL)

    puntos_auto = []    # polilínea base (verde)
    puntos_ext = []     # solo extensión manual (azul)
    img_original = None
    img_display = None

    # mouse
    def click_event(event, x, y, flags, param):
        nonlocal puntos_ext, img_display
        if event == cv2.EVENT_LBUTTONDOWN:
            puntos_ext.append((x, y))
            img_display = img_original.copy()
            _dibujar(img_display, puntos_auto, puntos_ext)

    cv2.setMouseCallback("Calibración Manual", click_event)

    # ------------ Loop por imágenes ------------
    for nombre in sorted(imagenes):
        puntos_auto = []
        puntos_ext = []

        ruta = os.path.join(carpeta_recortes, nombre)
        img_original = cv2.imread(ruta)
        if img_original is None:
            print(f"⚠️ No se pudo abrir {nombre}")
            continue

        # Paso 1: detección automática -> polilínea VERDE (muestreada como te gustaba)
        long_base_px = 0.0
        try:
            long_px, img_auto = detectar_radicula(img_original)
            if long_px > 0:
                # recuperar la línea azul del auto y muestrear (para que se vea igual que antes)
                mask_blue = cv2.inRange(img_auto, (255, 0, 0), (255, 0, 0))
                coords = np.column_stack(np.where(mask_blue > 0))  # (y,x)
                if len(coords) > 0:
                    crudos = [tuple(reversed(c)) for c in coords]  # (x,y)
                    step = max(1, len(crudos)//30)                 # mismo muestreo que tenías
                    puntos_auto = crudos[::step]
                    #long_base_px = _dist_total(puntos_auto)
                    long_base_px = float(long_px)
                    print(f"🔹 {nombre}: base automática = {round(long_base_px,2)} px ({len(puntos_auto)} pts)")
                else:
                    print(f"🔸 {nombre}: no se pudieron recuperar puntos automáticos.")
            else:
                print(f"🔸 {nombre}: sin detección automática (podés trazar manual).")
        except Exception as e:
            print(f"⚠️ Error en detección automática de {nombre}: {e}")

        # Paso 2: mostrar base y permitir extender
        img_display = img_original.copy()
        _dibujar(img_display, puntos_auto, puntos_ext)

        while True:
            cv2.imshow("Calibración Manual", img_display)
            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # ENTER -> confirmar
                salida = os.path.join(out_dir, f"ajustada_{nombre}")

                if puntos_auto and puntos_ext:
                    cadena_ext = [puntos_auto[-1]] + puntos_ext
                    long_ext_px = _dist_total(cadena_ext)
                elif (not puntos_auto) and len(puntos_ext) > 1:
                    long_ext_px = _dist_total(puntos_ext)
                else:
                    long_ext_px = 0.0

                long_total_px = long_base_px + long_ext_px

                if long_total_px > 0:
                    resultados.append([
                        nombre,
                        round(long_total_px, 2),
                        round(long_total_px * PX_A_MM, 2),
                        round(long_total_px * PX_A_CM, 2),
                        "GERMINADA"
                    ])
                    final_viz = img_original.copy()
                    _dibujar(final_viz, puntos_auto, puntos_ext)
                    cv2.imwrite(salida, final_viz)
                    print(f"✅ {nombre}: BASE {round(long_base_px,2)} px + EXT {round(long_ext_px,2)} px = TOTAL {round(long_total_px,2)} px")
                else:
                    resultados.append([nombre, "N/G", "N/G", "N/G", "NO GERMINADA"])
                    cv2.imwrite(salida, img_original)
                    print(f"⚠️ {nombre}: sin longitud válida, marcado N/G (guardado igual)")
                break

            elif key == ord('r'):  # r -> reiniciar dibujo (borrar todo)
                puntos_auto.clear()
                long_base_px = 0.0
                puntos_ext.clear()
                img_display = img_original.copy()
                print("🔁 Reiniciado: se borraron todos los puntos. Podés volver a dibujar desde cero.")


            elif key == ord('R'):  # Shift + R -> descartar totalmente (N/G)
                puntos_auto.clear()
                puntos_ext.clear()
                img_display = img_original.copy()
                resultados.append([nombre, "N/G", "N/G", "N/G", "NO GERMINADA"])
                salida = os.path.join(out_dir, f"ajustada_{nombre}")
                cv2.imwrite(salida, img_original)
                print(f"⚠️ {nombre}: descartada, marcada como N/G (guardado igual)")
                break

            elif key in [ord('z'), ord('Z')]:
                if puntos_ext:
                    puntos_ext.pop()
                    img_display = img_original.copy()
                    _dibujar(img_display, puntos_auto, puntos_ext)
                    print("↩️ Último punto de la extensión eliminado.")
                else:
                    print("⚠️ No hay puntos de extensión para eliminar.")

            elif key == 27:  # ESC
                print("🚪 Saliendo del modo manual...")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Celda", "Longitud_px", "Longitud_mm", "Longitud_cm", "Estado"])
                    writer.writerows(resultados)
                cv2.destroyAllWindows()
                return

        # pequeño refresh visual entre imágenes (sin cerrar ventana)
        img_display[:] = 255
        cv2.imshow("Calibración Manual", img_display)
        cv2.waitKey(80)

    # === Guardar CSV al final ===
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Celda", "Longitud_px", "Longitud_mm", "Longitud_cm", "Estado"])
        writer.writerows(resultados)

    print(f"\n📄 CSV guardado en: {csv_path}")
    cv2.destroyAllWindows()

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
    Permite restaurar una imagen calibrada y extender el trazo azul o reiniciar manualmente con R.
    - Clicks sucesivos: agrega puntos.
    - R: reinicia trazado manual (borra detección previa).
    - Z: borra último punto.
    - ENTER: guarda y suma longitud nueva al CSV.
    - ESC: salir sin guardar.
    """

    img_calibrada = cv2.imread(path_resultado)
    if img_calibrada is None:
        print(f"❌ No se pudo abrir la imagen calibrada:\n{path_resultado}")
        return None

    # === Detectar último punto azul si ya existe un trazo previo ===
    def detectar_ultimo_punto_azul(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 120, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        # punto más abajo (mayor Y)
        ultimo_punto = tuple(coords[coords[:, :, 1].argmax()][0])
        return ultimo_punto

    ultimo_punto = detectar_ultimo_punto_azul(img_calibrada)

    # === Detectar número de celda ===
    nombre_archivo = os.path.basename(path_resultado)
    match = re.search(r"celda_\d+", nombre_archivo)
    if not match:
        print(f"⚠️ No se pudo identificar el número de celda en: {nombre_archivo}")
        return None
    nombre_celda = match.group(0) + ".jpg"

    # === Construir ruta hacia la versión limpia ===
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

    puntos = []
    punto_inicial = ultimo_punto
    modo_manual = False

    print("🖋 Teclas disponibles → [R] reiniciar trazado, [Z] borrar punto, [ENTER] guardar, [ESC] salir.")
    print(f"🧠 Continuando desde: {punto_inicial if punto_inicial else '(sin trazo previo)'}")

    # === Función para calcular longitud total ===
    def calcular_longitud(pts):
        total = 0
        for i in range(1, len(pts)):
            total += math.dist(pts[i - 1], pts[i])
        return total

    # === Dibujar con longitud en tiempo real ===
    def redibujar(base_img):
        nonlocal img_display
        img_display = base_img.copy()

        # Redibujar líneas y puntos
        for i in range(1, len(puntos)):
            cv2.line(img_display, puntos[i - 1], puntos[i], (255, 0, 0), 1)
        for p in puntos:
            cv2.circle(img_display, p, 1, (255, 0, 0), -1)

        # Calcular longitud total (en cm)
        if len(puntos) > 1:
            total_px = calcular_longitud(puntos)
            total_cm = total_px * PX_A_CM
            texto = f"Longitud: {total_cm:.2f} cm"

            overlay = img_display.copy()
            cv2.rectangle(overlay, (5, 5), (300, 35), (0, 0, 0), -1)
            img_display = cv2.addWeighted(overlay, 0.5, img_display, 0.5, 0)
            cv2.putText(
                img_display, texto, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
            )

        cv2.imshow("Calibración Manual", img_display)

    # === Callback de clicks ===
    def click_event(event, x, y, flags, param):
        nonlocal puntos, punto_inicial, modo_manual
        if event == cv2.EVENT_LBUTTONDOWN:
            nuevo_punto = (x, y)

            # Si es el primer clic y hay un trazo previo, arrancar desde ese punto
            if not puntos and not modo_manual and punto_inicial is not None:
                puntos.append(punto_inicial)

            puntos.append(nuevo_punto)
            punto_inicial = nuevo_punto

            base = img_original if modo_manual else img_calibrada
            redibujar(base)

    # === Ventana principal ===
    cv2.namedWindow("Calibración Manual", cv2.WINDOW_NORMAL)
    cv2.imshow("Calibración Manual", img_display)
    cv2.setMouseCallback("Calibración Manual", click_event)

    csv_path = None
    punto_codo = None

    while True:
        key = cv2.waitKey(0) & 0xFF
         # Si el usuario cierra la ventana de OpenCV manualmente
        if cv2.getWindowProperty("Calibración Manual", cv2.WND_PROP_VISIBLE) < 1:
            print("🚪 Ventana cerrada manualmente — cancelado sin guardar cambios.")
            break

        # 🔹 R: modo manual desde imagen limpia
        if key in [ord('r'), ord('R')]:
            modo_manual = True
            puntos.clear()
            punto_inicial = None
            img_display = img_original.copy()
            cv2.imshow("Calibración Manual", img_display)
            print("🔄 Reinicio manual: haga clics para trazar desde cero.")

        # 🔹 Z: borrar último punto
        elif key in [ord('z'), ord('Z')]:
            if puntos:
                puntos.pop()
                base = img_original if modo_manual else img_calibrada
                redibujar(base)
                print("↩ Último punto eliminado.")
            else:
                print("⚠ No hay puntos para borrar.")

        # 🔹 ENTER: guardar trazado y actualizar CSV
        elif key == 13:  # ENTER
            if len(puntos) > 1:
                long_nueva_px = calcular_longitud(puntos)
                long_nueva_mm = long_nueva_px * PX_A_MM
                long_nueva_cm = long_nueva_px * PX_A_CM
                print(f"➕ Longitud nueva: {long_nueva_cm:.2f} cm")

                base = img_original if modo_manual else img_calibrada
                img_final = base.copy()
                for i in range(1, len(puntos)):
                    cv2.line(img_final, puntos[i - 1], puntos[i], (255, 0, 0), 1)
                for p in puntos:
                    cv2.circle(img_final, p, 1, (255, 0, 0), -1)
                cv2.imwrite(path_resultado, img_final)
                print(f"✅ Imagen actualizada: {path_resultado}")

                # === Actualizar CSV sumando longitud nueva ===
                csv_files = [f for f in os.listdir(carpeta_resultados) if f.lower().endswith(".csv")]
                if csv_files:
                    csv_path = os.path.join(carpeta_resultados, csv_files[0])
                    try:
                        df = pd.read_csv(csv_path)
                        mask = df["Celda"].astype(str).str.contains(match.group(0), case=False, regex=True)

                        # 🔧 Asegurar columnas numéricas
                        for col in ["Longitud_px", "Longitud_mm", "Longitud_cm"]:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                        # 🔧 Sumar nueva longitud
                        prev_val = df.loc[mask, "Longitud_px"].fillna(0)
                        df.loc[mask, "Longitud_px"] = round(prev_val + long_nueva_px, 2)
                        df.loc[mask, "Longitud_mm"] = round(df.loc[mask, "Longitud_px"] * PX_A_MM, 2)
                        df.loc[mask, "Longitud_cm"] = round(df.loc[mask, "Longitud_px"] * PX_A_CM, 2)
                        df.loc[mask, "Estado"] = "GERMINADA"

                        df.to_csv(csv_path, index=False)
                        print(f"📄 CSV actualizado → total {df.loc[mask, 'Longitud_cm'].values[0]} cm")

                    except Exception as e:
                        print(f"❌ Error al actualizar CSV: {e}")

            else:
                # Si no hay puntos trazados, marcar como NO GERMINADA
                shutil.copy(ruta_original, path_resultado)
                csv_files = [f for f in os.listdir(carpeta_resultados) if f.lower().endswith(".csv")]
                if csv_files:
                    csv_path = os.path.join(carpeta_resultados, csv_files[0])
                    try:
                        df = pd.read_csv(csv_path)
                        mask = df["Celda"].astype(str).str.contains(match.group(0), case=False, regex=True)
                        df.loc[mask, ["Longitud_px", "Longitud_mm", "Longitud_cm"]] = "N/G"
                        df.loc[mask, "Estado"] = "NO GERMINADA"
                        df.to_csv(csv_path, index=False)
                        print("📄 CSV actualizado como NO GERMINADA")
                    except Exception as e:
                        print(f"❌ Error al actualizar CSV: {e}")
            break
        # 🔹 ESC: salir sin guardar
        elif key in [27, ord('q')]:  # ESC o Q
            print("🚪 Cancelado sin guardar cambios.")
            break

    cv2.destroyAllWindows()
    return csv_path