import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuración de la ventana principal
ctk.set_appearance_mode("Dark")  # Modo oscuro
ctk.set_default_color_theme("blue")  # Tema azul

class RetenedorOrden0App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Retenedor de Orden 0 - ZOH")
        self.geometry("800x600")

        # Variables
        self.freq = ctk.DoubleVar(value=1.0)  # Frecuencia inicial

        # Crear slider para modificar la frecuencia
        self.slider_label = ctk.CTkLabel(self, text="Frecuencia:")
        self.slider_label.pack(pady=5)

        self.slider = ctk.CTkSlider(self, from_=0.5, to=10, variable=self.freq, command=self.actualizar_grafica)
        self.slider.pack(pady=5)

        # Frame para la gráfica
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)

        # Botón para actualizar manualmente
        self.update_button = ctk.CTkButton(self, text="Actualizar Gráfica", command=self.actualizar_grafica)
        self.update_button.pack(pady=5)

        # Mostrar la gráfica inicial
        self.actualizar_grafica()

    def retenedor_orden_0(self, t, signal):
        """
        Aplica un retenedor de orden 0 a la señal.
        """
        n = len(signal)
        zoh_signal = np.repeat(signal, 2)[1:]  # Duplicar valores para efecto de escalón
        t_zoh = np.repeat(t, 2)[1:]  # Mismo tiempo repetido para el escalón
        return t_zoh, zoh_signal

    def actualizar_grafica(self, event=None):
        """
        Actualiza la gráfica con la frecuencia seleccionada.
        """
        freq = self.freq.get()
        t = np.linspace(0, 1, 100)  # 1 segundo de duración
        signal = np.sin(2 * np.pi * freq * t)  # Señal senoidal

        # Aplicar retenedor de orden 0
        t_zoh, signal_zoh = self.retenedor_orden_0(t, signal)

        # Imprimir información en la terminal
        print("\n===== Actualización de la Gráfica =====")
        print(f"Frecuencia seleccionada: {freq:.2f} Hz")
        print(f"Número de muestras: {len(t)}")

        # Graficar
        self.ax.clear()
        self.ax.plot(t, signal, label="Señal Original", linestyle="dashed", color="blue")
        self.ax.step(t_zoh, signal_zoh, label="Retenedor Orden 0", color="red", where="post")
        self.ax.set_title("Retenedor de Orden 0 (ZOH)")
        self.ax.set_xlabel("Tiempo (s)")
        self.ax.set_ylabel("Amplitud")
        self.ax.legend()
        self.ax.grid(True, linestyle="--", alpha=0.6)

        self.canvas.draw()

# Ejecutar la aplicación
if __name__ == "__main__":
    app = RetenedorOrden0App()
    app.mainloop()
