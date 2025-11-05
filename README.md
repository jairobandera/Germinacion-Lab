
🌱 GerminIA Lab

GerminIA Lab es una aplicación de escritorio desarrollada en Python + Tkinter + OpenCV + Pandas, diseñada para automatizar el análisis de ensayos de germinación.
El sistema detecta radículas (raíces embrionarias) en celdas de placas de Petri, mide su longitud en píxeles y milímetros, y genera reportes en CSV/Excel con imágenes procesadas y clasificaciones automáticas o manuales.

## 🧠 Características principales

 📸 Gestión visual de imágenes: subida, vista de pendientes y resultados.

## 🧩 Procesamiento automático:

Recorte de placas y celdas.

Detección automática de germinación mediante filtros binarios y esqueletización.

Generación de resultados CSV y miniaturas etiquetadas.

## ✏️ Calibración manual asistida:

Dibujo y edición del trazo radicular con clicks del mouse.

Marcado de “codo” para separar radícula e hipocótilo.

## 📊 Visualización de resultados:

Miniaturas interactivas con sus longitudes medidas.

Tabla resumen con valores numéricos y exportación a Excel.

🧠 Modo automático o manual (seleccionable desde la interfaz).

## 🧱 Estructura del proyecto

```bash
Germinacion-Lab/
├── ui/app.py                # Interfaz gráfica principal (Tkinter)
├── main.py                  # Punto de entrada de la app
├── core/detectar_radicula.py  # Lógica de detección y calibración de radículas
├── core/marcar_rectangulos.py # Dibujo de celdas sobre placas recortadas
├── core/recorte_placas.py     # Recorte automático de placas a partir de fotos originales
├── core/cortar_celdas.py      # Separación de cada celda individual
├── requirements.txt           # Dependencias del entorno virtual
└── data/
    └── germinacion/data/
        ├── originales/        # Imágenes subidas
        ├── procesadas/        # Placas y celdas recortadas
        └── resultados/        # CSV e imágenes analizadas

```

## ⚙️ Instalación

1) Clonar el repositorio:
```bash
git clone https://github.com/jairobandera/Germinacion-Lab.git
cd Germinacion-Lab
```

2) Crear entorno virtual

En Windows PowerShell:
```bash
python -m venv venv
venv\Scripts\activate
```

En Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

3) Instalar dependencias
pip install -r requirements.txt

## 🚀 Ejecución
```bash
python main.py
```

🖥️ Uso general

📂 Subir imágenes: selecciona las fotos originales de las placas.

⚙️ Procesar: ejecuta el procesamiento automático (recorte, detección, generación de resultados).

📊 Ver resultados: inspecciona las imágenes procesadas, longitudes y estados.

🖋 Calibrar manualmente: desde una miniatura, podés abrir la herramienta de calibración para ajustar o dibujar la radícula.

📤 Exportar a Excel: genera un archivo .xlsx o .csv con las longitudes y estados de germinación.

🧩 Controles de calibración manual

Durante la calibración de una imagen:
```bash
Tecla	Acción
🖱️ Click	Dibujar sobre el trazo (extender radícula)
R	Reemplazo total (dibujar desde cero)
Z	Deshacer último punto
C	Marcar “codo” (divide radícula e hipocótilo)
Enter	Guardar cambios
Esc / Q	Salir sin guardar
```

