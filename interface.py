import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from ultralytics import YOLO

class FruitDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección de Frutas")
        self.root.geometry("1200x700") # Tamaño inicial sugerido

        # --- Cargar YOLO ---
        self.model = YOLO('yolov8n.pt')
        self.last_detected_fruit = None
        self.last_detected_weight = None
        self.last_detected_price = None

        # --- Variables de Estado ---
        self.is_running = True
        self.cap = cv2.VideoCapture(0) # Inicializar la cámara
        self.total_var = tk.StringVar(value="TOTAL: $0.00")

        # --- Configuración del Layout Principal (Grid) ---

        # Columna 0: Video (Más grande, 60% del ancho)
        self.root.grid_columnconfigure(0, weight=6, minsize=700)
        # Columna 1: Info/Carrito (Mediana, 25% del ancho)
        self.root.grid_columnconfigure(1, weight=2, minsize=250)
        # Columna 2: Botones (Pequeña, 15% del ancho)
        self.root.grid_columnconfigure(2, weight=0, minsize=100)

        self.root.grid_rowconfigure(0, weight=1) # Fila principal se expande verticalmente

        # Estilos para los botones de acción
        self._setup_styles()

        # --- Creación de Paneles ---
        self._create_video_panel()
        self._create_info_panel()
        self._create_button_panel()

        # --- Iniciar la Cámara en un Hilo Separado ---
        if self.cap.isOpened():
            self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.video_thread.start()
        else:
            messagebox.showerror("Error de Cámara", "No se pudo abrir la cámara (cv2.VideoCapture(0)). Asegúrate de que esté disponible.")
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_styles(self):
        """Configura los estilos de los botones principales."""
        font_family = 'Arial'
        font_size = 18
        font_weight = 'bold'
        self.button_styles = {
        # Agregar fruta (verde)
            "add": {
                "bg": "#28a745", "fg": "white",
                "font": (font_family, font_size, font_weight),
                "width": 4, "height": 2,
                "relief": "raised", "bd": 3
            },
        # Remover fruta (gris oscuro)
            "remove": {
                "bg": "#6c757d", "fg": "white",
                "font": (font_family, font_size, font_weight),
                "width": 4, "height": 2,
                "relief": "raised", "bd": 3
            },
        # Obtener recibo (azul)
            "receipt": {
                "bg": "#007bff", "fg": "white",
                "font": (font_family, font_size, font_weight),
                "width": 4, "height": 2,
                "relief": "raised", "bd": 3
            },
        # Cancelar pedido (rojo)
            "cancel": {
                "bg": "#dc3545", "fg": "white",
                "font": (font_family, font_size, font_weight),
                "width": 4, "height": 2,
                "relief": "raised", "bd": 3
            }
        }

    def _create_video_panel(self):
        """Crea el panel de la cámara y la etiqueta de video."""
        self.video_panel = ttk.LabelFrame(self.root, text="Cámara y Detección en Tiempo Real")
        self.video_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_panel.grid_columnconfigure(0, weight=1)
        self.video_panel.grid_rowconfigure(0, weight=1)

        self.video_label = tk.Label(self.video_panel, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    def _create_info_panel(self):
        """Crea el panel de detalles y el carrito de compras."""
        self.info_panel = ttk.Frame(self.root)
        self.info_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Fila 0: Detalles (pequeño, weight=1) | Fila 1: Carrito (grande, weight=3)
        self.info_panel.grid_rowconfigure(0, weight=1)
        self.info_panel.grid_rowconfigure(1, weight=3)
        self.info_panel.grid_columnconfigure(0, weight=1)

        # --- Sub-Panel de Detalles de Producto ---
        self.details_frame = ttk.LabelFrame(self.info_panel, text="Detalles del Último Producto")
        self.details_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Labels para mostrar la información detectada (se actualizarán dinámicamente)
        self.lbl_fruta = ttk.Label(self.details_frame, text="Fruta: N/A", font=('Arial', 10))
        self.lbl_peso = ttk.Label(self.details_frame, text="Peso Estimado: N/A", font=('Arial', 10))
        self.lbl_precio_kg = ttk.Label(self.details_frame, text="Precio/kg: N/A", font=('Arial', 10))
        self.lbl_subtotal = ttk.Label(self.details_frame, text="Subtotal: N/A", font=('Arial', 10))

        self.lbl_fruta.pack(fill='x', padx=5, pady=2)
        self.lbl_peso.pack(fill='x', padx=5, pady=2)
        self.lbl_precio_kg.pack(fill='x', padx=5, pady=2)
        self.lbl_subtotal.pack(fill='x', padx=5, pady=2)

        # --- Sub-Panel de Recibo/Carrito ---
        self.receipt_frame = ttk.LabelFrame(self.info_panel, text="Carrito de Compras")
        self.receipt_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.receipt_frame.grid_rowconfigure(0, weight=1) # El treeview ocupa la mayor parte
        self.receipt_frame.grid_columnconfigure(0, weight=1)

        # Treeview para la lista de frutas (el "carrito")
        self.receipt_tree = ttk.Treeview(self.receipt_frame, columns=('Nombre', 'Peso', 'Precio'), show='headings')
        self.receipt_tree.heading('Nombre', text='Fruta')
        self.receipt_tree.heading('Peso', text='Peso (kg)')
        self.receipt_tree.heading('Precio', text='Total ($)')
        self.receipt_tree.column('Nombre', width=100, anchor='w')
        self.receipt_tree.column('Peso', width=70, anchor='center')
        self.receipt_tree.column('Precio', width=70, anchor='e')

        # Scrollbar vertical para el Treeview (carrito)
        vsb = ttk.Scrollbar(self.receipt_frame, orient="vertical", command=self.receipt_tree.yview)
        self.receipt_tree.configure(yscrollcommand=vsb.set)

        self.receipt_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky='ns')

        # Etiqueta de total final (parte inferior del carrito)
        ttk.Label(self.receipt_frame, textvariable=self.total_var,
                  font=('Arial', 14, 'bold'), foreground='navy').grid(row=1, column=0, columnspan=2, pady=5, sticky='e')

    def _create_button_panel(self):
        """Crea el panel con los botones de acción."""
        self.button_panel = tk.Frame(self.root)
        self.button_panel.grid(row=0, column=2, padx=10, pady=10, sticky="ns")

        padding_y = 20

        tk.Button(self.button_panel, text="➕", command=self.add_fruit,
              **self.button_styles["add"]).pack(pady=padding_y)

        tk.Button(self.button_panel, text="     🗑️", command=self.remove_fruit,
              **self.button_styles["remove"]).pack(pady=padding_y)

        tk.Button(self.button_panel, text="🧾", command=self.finalize_order,
              **self.button_styles["receipt"]).pack(pady=padding_y)

        tk.Button(self.button_panel, text="❌", command=self.cancel_order,
              **self.button_styles["cancel"]).pack(pady=padding_y)

    def _video_loop(self):
        """Maneja el stream de video de OpenCV y la actualización de Tkinter."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Pasar el frame a YOLO
                results = self.model(frame, verbose=False)[0]

                if len(results.boxes) > 0:
                    # Tomar la primera detección con mayor confianza
                    best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
                    class_id = int(best_box.cls[0])
                    class_name = results.names[class_id]
                    confidence = float(best_box.conf[0])

                    peso = 0.0180
                    precio_kg = 5.99
                    subtotal = peso * precio_kg

                    self.last_detected_fruit = class_name
                    self.last_detected_weight = peso
                    self.last_detected_price = subtotal

                    self.root.after(0, self._update_details,
                                    class_name,
                                    f'{peso:.3f} kg',
                                    f'${precio_kg:.2f}/kg',
                                    f'${subtotal:.2f}')

                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                width = self.video_label.winfo_width()
                height = self.video_label.winfo_height()

                if width > 0 and height > 0:
                    img_resized = img.resize((width, height))
                    img_tk = ImageTk.PhotoImage(image=img_resized)
                    self.video_label.imgtk = img_tk
                    self.video_label.configure(image=img_tk)

            time.sleep(0.03)

    def _update_details(self, fruta, peso, precio_kg, subtotal):
        """Actualiza los labels del panel de detalles (debe llamarse con root.after)."""
        self.lbl_fruta.config(text=f"Fruta: {fruta}")
        self.lbl_peso.config(text=f"Peso Estimado: {peso}")
        self.lbl_precio_kg.config(text=f"Precio/kg: {precio_kg}")
        self.lbl_subtotal.config(text=f"Subtotal: {subtotal}")

    # ----------------------------------------
    # --- Funciones de Botones (Acciones) ----
    # ----------------------------------------

    def add_fruit(self):
        """Añade la fruta actualmente detectada al carrito."""

        if self.last_detected_fruit:
            fruta_detectada = self.last_detected_fruit
            peso = self.last_detected_weight
            precio_total = self.last_detected_price

            self.receipt_tree.insert('', 'end', values=(fruta_detectada, f"{peso:.3f}", f"{precio_total:.2f}"))
            self._update_total()
            messagebox.showinfo("Carrito", f"{fruta_detectada} agregada.")
        else:
            messagebox.showwarning("Atención", "No se ha detectado ninguna fruta.")

    def remove_fruit(self):
        """Elimina la fruta seleccionada del carrito."""
        selected_item = self.receipt_tree.focus() # Obtiene el ID del ítem seleccionado
        if selected_item:
            self.receipt_tree.delete(selected_item)
            self._update_total()
            messagebox.showinfo("Carrito", "Elemento eliminado correctamente.")
        else:
            messagebox.showwarning("Atención", "Selecciona una fruta del carrito para eliminar.")

    def finalize_order(self):
        """Genera el recibo final y reinicia el carrito."""
        total = self._calculate_total()
        if total > 0:
            messagebox.showinfo("Recibo Final", f"Pedido completado. Total a pagar: ${total:.2f}")
            # Lógica para imprimir/guardar el recibo (Opcional)

            # Reiniciar el carrito
            for item in self.receipt_tree.get_children():
                self.receipt_tree.delete(item)
            self._update_total()
        else:
            messagebox.showwarning("Atención", "El carrito está vacío.")

    def cancel_order(self):
        """Cancela todo el pedido y vacía el carrito."""
        if messagebox.askyesno("Confirmar", "¿Está seguro que desea cancelar todo el pedido? Se vaciará el carrito."):
            for item in self.receipt_tree.get_children():
                self.receipt_tree.delete(item)
            self._update_total()
            messagebox.showinfo("Acción", "Pedido cancelado. Carrito vacío.")

    def _calculate_total(self):
        """Calcula la suma total de los precios en el Treeview."""
        total = 0.0
        for item in self.receipt_tree.get_children():
            values = self.receipt_tree.item(item, 'values')
            try:
                # El precio total es el valor en el índice 2
                total += float(values[2])
            except (ValueError, IndexError):
                pass # Ignorar filas con formato de precio incorrecto
        return total

    def _update_total(self):
        """Actualiza la etiqueta del total en el carrito."""
        total = self._calculate_total()
        self.total_var.set(f"TOTAL: ${total:.2f}")

    def on_closing(self):
        """Llamado al cerrar la ventana para liberar recursos."""
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

# --- Ejecución de la Aplicación ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FruitDetectorApp(root)
    root.mainloop()