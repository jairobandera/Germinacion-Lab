import tkinter as tk
from tkinter import ttk

# Paleta de colores (inspirada en Bootstrap)
COLORS = {
    "bg_main": "#f8f9fa",
    "bg_card": "#ffffff",
    "bg_sidebar": "#ffffff",
    "border": "#dee2e6",
    "accent": "#0d6efd",
    "accent_hover": "#0b5ed7",
    "text": "#212529",
    "muted": "#6c757d",
}

def apply_theme(root: tk.Tk):
    root.configure(bg=COLORS["bg_main"])
    style = ttk.Style(root)
    style.theme_use("clam")

    # Botones estilo Bootstrap
    style.configure("Accent.TButton",
                    background=COLORS["accent"],
                    foreground="white",
                    borderwidth=0,
                    padding=(8, 6),
                    font=("Segoe UI", 10, "bold"))
    style.map("Accent.TButton",
              background=[("active", COLORS["accent_hover"])])

    # Botones secundarios
    style.configure("Secondary.TButton",
                    background=COLORS["bg_card"],
                    foreground=COLORS["text"],
                    bordercolor=COLORS["border"],
                    relief="solid",
                    padding=(8, 6))
    style.map("Secondary.TButton",
              background=[("active", "#e9ecef")])

    # Labels generales
    style.configure("TLabel", background=COLORS["bg_main"], foreground=COLORS["text"])

    # Frame tipo Card
    style.configure("Card.TFrame",
                    background=COLORS["bg_card"],
                    relief="flat",
                    borderwidth=1)

    # Botón de peligro (Eliminar resultados)
    style.configure("Danger.TButton",
                    background="#dc3545",
                    foreground="white",
                    borderwidth=0,
                    padding=(8, 6),
                    font=("Segoe UI", 10, "bold"))
    style.map("Danger.TButton",
              background=[("active", "#bb2d3b"), ("disabled", "#e6a2a8")])

# Botón neutro (gris claro)
    style.configure("Neutral.TButton",
                    background="#f8f9fa",
                    foreground="#333",
                    borderwidth=1,
                    padding=(8, 6),
                    font=("Segoe UI", 10))
    style.map("Neutral.TButton",
              background=[("active", "#e2e6ea")])

    # Botón informativo (celeste)
    style.configure("Info.TButton",
                    background="#17a2b8",
                    foreground="white",
                    borderwidth=0,
                    padding=(8, 6),
                    font=("Segoe UI", 10, "bold"))
    style.map("Info.TButton",
              background=[("active", "#138496")])

    # Botón éxito (verde)
    style.configure("Success.TButton",
                    background="#28a745",
                    foreground="white",
                    borderwidth=0,
                    padding=(8, 6),
                    font=("Segoe UI", 10, "bold"))
    style.map("Success.TButton",
              background=[("active", "#218838")])

