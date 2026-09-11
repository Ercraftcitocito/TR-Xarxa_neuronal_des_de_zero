import sys
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

# Carregar pesos del model
try:
    model = np.load('weight&biases.npz')
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    print("Pesos carregats correctament.")
except FileNotFoundError:
    print("Error: No s'ha trobat 'weight&biases.npz'.")
    sys.exit()

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    A1 = ReLU(W1.dot(X) + b1)
    A2 = softmax(W2.dot(A1) + b2)
    return A2

def center_by_center_of_mass(img_array):
    total_mass = np.sum(img_array)
    if total_mass == 0:
        return img_array

    cy, cx = np.indices(img_array.shape)
    cy = np.sum(cy * img_array) / total_mass
    cx = np.sum(cx * img_array) / total_mass

    shift_y = round(13.5 - cy)
    shift_x = round(13.5 - cx)

    shifted = np.roll(img_array, shift_y, axis=0)
    shifted = np.roll(shifted, shift_x, axis=1)

    if shift_y > 0: shifted[:shift_y, :] = 0
    elif shift_y < 0: shifted[shift_y:, :] = 0
    if shift_x > 0: shifted[:, :shift_x] = 0
    elif shift_x < 0: shifted[:, shift_x:] = 0

    return shifted

class AppSimple:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Reconeixement de Dígits")

        # Escalat dinàmic basat en l'alçada de la pantalla (1080p com a base)
        screen_height = self.root.winfo_screenheight()
        scale = screen_height / 1080.0

        # Dimensions i fonts calculades proporcionalment
        self.CANVAS_SIZE = int(400 * scale)
        win_w = int(700 * scale)
        win_h = int(480 * scale)
        f_num = int(50 * scale)
        f_title = max(10, int(11 * scale))
        p_btn = int(8 * scale)

        self.root.geometry(f"{win_w}x{win_h}")
        self.root.resizable(False, False)
        self.root.configure(bg="#181825")

        # Imatge interna per a la xarxa
        self.image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        # Marges i posicions proporcionals
        pos_x = int(30 * scale)
        pos_y = int(40 * scale)
        panel_w = int(220 * scale)
        panel_x = self.CANVAS_SIZE + (pos_x * 2)

        # Canvas a l'esquerra
        self.canvas = tk.Canvas(
            self.root, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            bg="black", highlightthickness=2, highlightbackground="#45475a"
        )
        self.canvas.place(x=pos_x, y=pos_y)

        self.tk_img = ImageTk.PhotoImage(self.image)
        self.canvas_img = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        # Panell dret proporcional
        panel = tk.Frame(self.root, bg="#1e1e2e")
        panel.place(x=panel_x, y=pos_y, width=panel_w, height=self.CANVAS_SIZE)

        tk.Label(
            panel, text="PREDICCIÓ",
            font=("Arial", f_title, "bold"), fg="#89b4fa", bg="#1e1e2e"
        ).pack(pady=(int(25 * scale), int(5 * scale)))

        self.lbl_num = tk.Label(
            panel, text="-",
            font=("Helvetica", f_num, "bold"), fg="#a6e3a1", bg="#1e1e2e"
        )
        self.lbl_num.pack(pady=int(5 * scale))

        self.lbl_conf = tk.Label(
            panel, text="Dibuixa un número",
            font=("Helvetica", max(9, int(10 * scale))), fg="#cdd6f4", bg="#1e1e2e"
        )
        self.lbl_conf.pack(pady=int(5 * scale))

        btn_predecir = tk.Button(
            panel, text="Predir", command=self.predict,
            font=("Helvetica", f_title, "bold"), bg="#a6e3a1", fg="#11111b",
            relief="flat", cursor="hand2", pady=p_btn
        )
        btn_predecir.pack(fill="x", padx=int(20 * scale), pady=(int(20 * scale), int(10 * scale)))

        btn_limpiar = tk.Button(
            panel, text="Netejar", command=self.clear,
            font=("Helvetica", f_title, "bold"), bg="#f38ba8", fg="#11111b",
            relief="flat", cursor="hand2", pady=p_btn
        )
        btn_limpiar.pack(fill="x", padx=int(20 * scale), pady=int(5 * scale))

        # Esdeveniments de ratolí
        self.last_x, self.last_y = None, None
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        self.root.mainloop()

    def start_stroke(self, event):
        self.last_x, self.last_y = event.x, event.y

    def end_stroke(self, _event):
        self.last_x, self.last_y = None, None

    def draw_line(self, event):
        lx, ly = self.last_x, self.last_y
        if lx is not None and ly is not None:
            r = max(12, int(self.CANVAS_SIZE / 25))
            self.draw.line([lx, ly, event.x, event.y], fill=255, width=r * 2)
            self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=255)

            self.tk_img = ImageTk.PhotoImage(self.image)
            self.canvas.itemconfig(self.canvas_img, image=self.tk_img)

        self.last_x, self.last_y = event.x, event.y

    def clear(self):
        self.draw.rectangle((0, 0, self.CANVAS_SIZE, self.CANVAS_SIZE), fill=0)
        self.tk_img = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.canvas_img, image=self.tk_img)
        self.lbl_num.config(text="-")
        self.lbl_conf.config(text="Dibuja un número")

    def predict(self):
        bbox = self.image.getbbox()
        if bbox is None:
            self.lbl_conf.config(text="Canvas buit")
            return

        img_crop = self.image.crop(bbox)
        img_crop.thumbnail((20, 20), Image.Resampling.LANCZOS)

        img_28 = Image.new("L", (28, 28), color=0)
        px, py = (28 - img_crop.width) // 2, (28 - img_crop.height) // 2
        img_28.paste(img_crop, (px, py))

        img_array = np.array(img_28, dtype=np.float32) / 255.0
        img_array = center_by_center_of_mass(img_array)

        X_in = img_array.flatten().reshape(784, 1)
        A2 = forward_prop(W1, b1, W2, b2, X_in)

        pred = np.argmax(A2, axis=0)[0]
        conf = np.max(A2, axis=0)[0] * 100

        self.lbl_num.config(text=str(pred))
        self.lbl_conf.config(text=f"Confiança: {conf:.1f}%")

if __name__ == "__main__":
    AppSimple()
