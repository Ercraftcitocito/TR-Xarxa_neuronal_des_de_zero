import sys
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw

# 1. Cargar el modelo entrenado
try:
    modelo = np.load('pesos_modelo.npz')
    W1, b1, W2, b2 = modelo['W1'], modelo['b1'], modelo['W2'], modelo['b2']
    print("Pesos cargados correctamente.")
except FileNotFoundError:
    print("Error: No se encontró 'pesos_modelo.npz'. Ejecuta el entrenamiento primero.")
    sys.exit()

# 2. Funciones de predicción de la red neuronal
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    A = np.exp(Z)
    A = A / np.sum(A, axis=0, keepdims=True)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# 3. Interfaz Gráfica con Tkinter
class AppDibujo:
    def __init__(self):
        self.ventana = tk.Tk()
        self.ventana.title("Reconocimiento de Dígitos MNIST - Ajustado")

        # Lienzo más grande para dibujar cómodamente (400x400)
        self.canvas_width = 400
        self.canvas_height = 400

        self.canvas = tk.Canvas(self.ventana, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack(pady=10)

        # Imagen PIL en memoria
        self.imagen = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.imagen)

        # Evento de dibujo con el ratón
        self.canvas.bind("<B1-Motion>", self.dibujar)

        # Botones
        frame_botones = tk.Frame(self.ventana)
        frame_botones.pack()

        btn_predecir = tk.Button(frame_botones, text="Predecir", command=self.hacer_prediccion, font=("Helvetica", 12))
        btn_predecir.pack(side=tk.LEFT, padx=10, pady=5)

        btn_limpiar = tk.Button(frame_botones, text="Limpiar", command=self.limpiar_canvas, font=("Helvetica", 12))
        btn_limpiar.pack(side=tk.LEFT, padx=10, pady=5)

        self.etiqueta_resultado = tk.Label(self.ventana, text="Dibuja un número y pulsa Predecir", font=("Helvetica", 14))
        self.etiqueta_resultado.pack(pady=15)

        self.ventana.mainloop()

    def dibujar(self, event):
        x, y = event.x, event.y
        r = 15  # Pincel más grueso adaptado al nuevo tamaño del lienzo
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def limpiar_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, self.canvas_width, self.canvas_height), fill=0)
        self.etiqueta_resultado.config(text="Lienzo limpio")

    def hacer_prediccion(self):
        # Procesamiento inteligente estilo MNIST:
        # 1. Recortar los bordes negros sobrantes alrededor del dibujo
        bbox = self.imagen.getbbox()
        if bbox is None:
            self.etiqueta_resultado.config(text="¡Dibuja algo primero!")
            return

        imagen_recortada = self.imagen.crop(bbox)

        # 2. Mantener la proporción y redimensionar a un recuadro de 20x20
        imagen_recortada.thumbnail((20, 20), Image.Resampling.LANCZOS)

        # 3. Crear una imagen nueva de 28x28 con fondo negro y pegar el número perfectamente centrado
        imagen_final = Image.new("L", (28, 28), color=0)
        paste_x = (28 - imagen_recortada.width) // 2
        paste_y = (28 - imagen_recortada.height) // 2
        imagen_final.paste(imagen_recortada, (paste_x, paste_y))

        # 4. Convertir a matriz, normalizar y aplanar
        img_array = np.array(imagen_final) / 255.0
        X_input = img_array.flatten().reshape(784, 1)

        # 5. Predicción con la red neuronal
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_input)

        prediccion = np.argmax(A2, 0)[0]
        confianza = np.max(A2, 0)[0] * 100

        self.etiqueta_resultado.config(text=f"Predicción: {prediccion} (Confianza: {confianza:.1f}%)")

if __name__ == "__main__":
    AppDibujo()
