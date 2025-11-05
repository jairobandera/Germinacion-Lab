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
