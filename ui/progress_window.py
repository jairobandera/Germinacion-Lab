import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import sys
import threading
import io


class ConsoleRedirect(io.StringIO):
    """Redirige print() a la ventana Tkinter"""
    def __init__(self, text_widget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass


class ProgressWindow(tk.Toplevel):
    def __init__(self, parent, title="Progreso del procesamiento"):
        super().__init__(parent)
        self.title(title)
        self.geometry("700x400")
        self.configure(bg="#f4f4f4")
        self.text = ScrolledText(self, wrap=tk.WORD, font=("Consolas", 10))
        self.text.pack(expand=True, fill="both", padx=10, pady=10)
        self.text.configure(state='disabled')

        self.protocol("WM_DELETE_WINDOW", self.hide)
        self.withdraw()  # oculta al inicio

        self._stdout_backup = None
        self._stderr_backup = None

    def show(self):
        self.deiconify()
        self.lift()
        self._stdout_backup = sys.stdout
        self._stderr_backup = sys.stderr
        sys.stdout = ConsoleRedirect(self.text)
        sys.stderr = ConsoleRedirect(self.text)

    def hide(self):
        sys.stdout = self._stdout_backup
        sys.stderr = self._stderr_backup
        self.withdraw()

    def run_in_thread(self, target):
        """Ejecuta la función target en un hilo para no congelar la UI"""
        hilo = threading.Thread(target=self._run_and_close, args=(target,), daemon=True)
        hilo.start()

    def _run_and_close(self, target):
        try:
            target()
        finally:
            print("\n✅ Proceso finalizado.")
            self.hide()
