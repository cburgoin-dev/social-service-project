import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class FruitDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección de Frutas")
        self.root.geometry("1200x700")

        # --- Cargar tu modelo de frutas entrenado ---
        self.model = load_model("fruits360_mobilenetv2_finetuned(alternative).h5")
        #self.model = load_model("fruits_real_optimized.h5")

        # --- Cargar las clases desde el dataset original ---
        data_dir = "C:/Users/Emili/Downloads/archive (1)/train"  # ⚠️ Ajusta esta ruta a tu dataset, el que les voy a jitubear
        #data_dir = "C:/Users/Emili/Downloads/fruits_data_set/Training"  # ⚠️ Ajusta esta ruta a tu dataset
        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=(100, 100),
            batch_size=1,
            class_mode='categorical'
        )
        self.class_names = list(generator.class_indices.keys())

        # --- Variables de Estado ---
        self.last_detected_fruit = None
        self.last_detected_weight = None
        self.last_detected_price = None
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.total_var = tk.StringVar(value="TOTAL: $0.00")

        # --- Layout principal ---
        self._setup_styles()
        self._create_video_panel()
        self._create_info_panel()
        self._create_button_panel()

        # --- Iniciar la cámara ---
        if self.cap.isOpened():
            self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.video_thread.start()
        else:
            messagebox.showerror("Error", "No se pudo abrir la cámara.")
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # -----------------------------------------------
    # Estilos y Layout
    # -----------------------------------------------
    def _setup_styles(self):
        font_family = 'Arial'
        font_size = 18
        font_weight = 'bold'
        self.button_styles = {
            "add": {"bg": "#28a745", "fg": "white", "font": (font_family, font_size, font_weight),
                    "width": 4, "height": 2, "relief": "raised", "bd": 3},
            "remove": {"bg": "#6c757d", "fg": "white", "font": (font_family, font_size, font_weight),
                       "width": 4, "height": 2, "relief": "raised", "bd": 3},
            "receipt": {"bg": "#007bff", "fg": "white", "font": (font_family, font_size, font_weight),
                        "width": 4, "height": 2, "relief": "raised", "bd": 3},
            "cancel": {"bg": "#dc3545", "fg": "white", "font": (font_family, font_size, font_weight),
                       "width": 4, "height": 2, "relief": "raised", "bd": 3}
        }

    def _create_video_panel(self):
        self.video_panel = ttk.LabelFrame(self.root, text="Cámara y Detección en Tiempo Real")
        self.video_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_label = tk.Label(self.video_panel, bg="black")
        self.video_label.pack(fill="both", expand=True)

    def _create_info_panel(self):
        self.info_panel = ttk.Frame(self.root)
        self.info_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Panel de detalles
        self.details_frame = ttk.LabelFrame(self.info_panel, text="Detalles del Último Producto")
        self.details_frame.pack(fill="x", pady=5)

        self.lbl_fruta = ttk.Label(self.details_frame, text="Fruta: N/A")
        self.lbl_peso = ttk.Label(self.details_frame, text="Peso Estimado: N/A")
        self.lbl_precio_kg = ttk.Label(self.details_frame, text="Precio/kg: N/A")
        self.lbl_subtotal = ttk.Label(self.details_frame, text="Subtotal: N/A")

        for lbl in [self.lbl_fruta, self.lbl_peso, self.lbl_precio_kg, self.lbl_subtotal]:
            lbl.pack(anchor="w", padx=5, pady=2)

        # Panel del carrito
        self.receipt_frame = ttk.LabelFrame(self.info_panel, text="Carrito de Compras")
        self.receipt_frame.pack(fill="both", expand=True, pady=5)

        self.receipt_tree = ttk.Treeview(self.receipt_frame, columns=('Nombre', 'Peso', 'Precio'), show='headings')
        self.receipt_tree.heading('Nombre', text='Fruta')
        self.receipt_tree.heading('Peso', text='Peso (kg)')
        self.receipt_tree.heading('Precio', text='Total ($)')
        self.receipt_tree.pack(fill="both", expand=True)

        ttk.Label(self.receipt_frame, textvariable=self.total_var,
                  font=('Arial', 14, 'bold'), foreground='navy').pack(anchor="e", pady=5)

    def _create_button_panel(self):
        self.button_panel = tk.Frame(self.root)
        self.button_panel.grid(row=0, column=2, padx=10, pady=10, sticky="ns")

        for text, cmd, style in [
            ("➕", self.add_fruit, "add"),
            ("🗑️", self.remove_fruit, "remove"),
            ("🧾", self.finalize_order, "receipt"),
            ("❌", self.cancel_order, "cancel")
        ]:
            tk.Button(self.button_panel, text=text, command=cmd, **self.button_styles[style]).pack(pady=20)

    # -----------------------------------------------
    # Procesamiento de video
    # -----------------------------------------------
    def preprocess_frame(self, frame):
        # Convertir a RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 🔹 Eliminar fondo claro (blanco, gris claro)
        mask = cv2.inRange(img_rgb, (180, 180, 180), (255, 255, 255))
        img_rgb[mask == 255] = (255, 255, 255)
        
        # 🔹 Mejorar contraste (útil para cámara)
        img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=10)
        
        # 🔹 Redimensionar y normalizar
        img_resized = cv2.resize(img_rgb, (100, 100))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        return img_array


    def _video_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                img_array = self.preprocess_frame(frame)
                preds = self.model.predict(img_array, verbose=0)
                class_name = self.class_names[np.argmax(preds)]

                # Simulación de peso/precio (puedes cambiar según tu lógica real)
                peso = 0.2
                precio_kg = 8.99
                subtotal = peso * precio_kg

                # Guardar detección actual
                self.last_detected_fruit = class_name
                self.last_detected_weight = peso
                self.last_detected_price = subtotal

                self.root.after(0, self._update_details,
                                class_name,
                                f'{peso:.3f} kg',
                                f'${precio_kg:.2f}/kg',
                                f'${subtotal:.2f}')

                # Mostrar el frame en Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img.resize((640, 480)))
                self.video_label.imgtk = img_tk
                self.video_label.configure(image=img_tk)
            time.sleep(0.05)

    # -----------------------------------------------
    # Actualización de información
    # -----------------------------------------------
    def _update_details(self, fruta, peso, precio_kg, subtotal):
        self.lbl_fruta.config(text=f"Fruta: {fruta}")
        self.lbl_peso.config(text=f"Peso Estimado: {peso}")
        self.lbl_precio_kg.config(text=f"Precio/kg: {precio_kg}")
        self.lbl_subtotal.config(text=f"Subtotal: {subtotal}")

    # -----------------------------------------------
    # Botones
    # -----------------------------------------------
    def add_fruit(self):
        if self.last_detected_fruit:
            fruta = self.last_detected_fruit
            peso = self.last_detected_weight
            precio_total = self.last_detected_price
            self.receipt_tree.insert('', 'end', values=(fruta, f"{peso:.3f}", f"{precio_total:.2f}"))
            self._update_total()
            messagebox.showinfo("Carrito", f"{fruta} agregada.")
        else:
            messagebox.showwarning("Atención", "No se ha detectado ninguna fruta.")

    def remove_fruit(self):
        selected = self.receipt_tree.focus()
        if selected:
            self.receipt_tree.delete(selected)
            self._update_total()
        else:
            messagebox.showwarning("Atención", "Selecciona una fruta para eliminar.")

    def finalize_order(self):
        total = self._calculate_total()
        if total > 0:
            messagebox.showinfo("Recibo", f"Total a pagar: ${total:.2f}")
            for i in self.receipt_tree.get_children():
                self.receipt_tree.delete(i)
            self._update_total()
        else:
            messagebox.showwarning("Atención", "El carrito está vacío.")

    def cancel_order(self):
        for i in self.receipt_tree.get_children():
            self.receipt_tree.delete(i)
        self._update_total()

    def _calculate_total(self):
        total = 0
        for item in self.receipt_tree.get_children():
            total += float(self.receipt_tree.item(item, 'values')[2])
        return total

    def _update_total(self):
        total = self._calculate_total()
        self.total_var.set(f"TOTAL: ${total:.2f}")

    # -----------------------------------------------
    def on_closing(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

# -----------------------------------------------
# Ejecución
# -----------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FruitDetectorApp(root)
    root.mainloop()
