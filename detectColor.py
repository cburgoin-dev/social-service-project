import cv2 as cv
import tkinter
import numpy as np

window = tkinter.Tk()
window.geometry("700x700")

def start():
    print("Inicio")
    window.destroy()

# Fondo primero
img = tkinter.PhotoImage(file="C:/Users/Crist/OneDrive/Desktop/software-development/projects/social-service-project/fondo.png")
lbl_img = tkinter.Label(window, image=img)
lbl_img.place(x=0, y=0, relwidth=1, relheight=1)

# Título
title = tkinter.Label(window, text="Carrot Vision", font=("Arial", 20))
title.place(relx=0.5, rely=0.3, anchor="center")

# Botón en el centro
btn_start = tkinter.Button(window, text="Empezar Escaneo de Frutas", font=("Arial", 16), command=start)
btn_start.place(relx=0.5, rely=0.5, anchor="center")

window.mainloop()

def show_menu(frame):
    '''Dibujar instrucciones en pantalla.'''
    menu = [
        "1. Rojo",
        "2. Verde",
        "3. Azul",
        "4. Amarillo",
        "5. Naranja",
        "6. Morado",
        "7. Rosa",
        "8. Cafe",
        "ESC - Exit"
    ]
    y0, dy = 30, 35  # posición inicial y separación
    for i, line in enumerate(menu):
        y = y0 + i*dy
        cv.putText(frame, line, (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 0, 0), 2, cv.LINE_AA)

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

cap = cv.VideoCapture(1) # Cámara de video adicional

colors_bgr = {
        "Rojo":    (0, 0, 255),
        "Verde":  (0, 255, 0),
        "Azul":   (255, 0, 0),
        "Amarillo": (0, 255, 255),
        "Naranja": (0, 165, 255),
        "Morado": (128, 0, 128),
        "Rosa":   (203, 192, 255),
        "Cafe":  (42, 42, 165)
    }

current_mask = 'Rojo'  # máscara inicial
current_dominant = 'Rojo' # color dominante inicial

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Rango de colores
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 255, 255])

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([20, 255, 255])

    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])

    lower_pink = np.array([160, 100, 100])
    upper_pink = np.array([170, 255, 255])

    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([20, 255, 200])

    # Máscaras
    red_mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(red_mask1, red_mask2)

    green_mask = cv.inRange(hsv, lower_green, upper_green)
    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    orange_mask = cv.inRange(hsv, lower_orange, upper_orange)
    purple_mask = cv.inRange(hsv, lower_purple, upper_purple)
    pink_mask = cv.inRange(hsv, lower_pink, upper_pink)
    brown_mask = cv.inRange(hsv, lower_brown, upper_brown)

    # Selección de máscara según variable
    if current_mask == 'Rojo':
        mask = red_mask
    elif current_mask == 'Verde':
        mask = green_mask
    elif current_mask == 'Azul':
        mask = blue_mask
    elif current_mask == 'Amarillo':
        mask = yellow_mask
    elif current_mask == 'Naranja':
        mask = orange_mask
    elif current_mask == 'Morado':
        mask = purple_mask
    elif current_mask == 'Rosa':
        mask = pink_mask
    elif current_mask == 'Cafe':
        mask = brown_mask
    res = cv.bitwise_and(frame, frame, mask=mask)

    # --- Mostrar menú en pantalla ---
    show_menu(frame)
    cv.putText(frame, f'Mascara Actual: {current_mask}',
               (10, 365), cv.FONT_HERSHEY_SIMPLEX, 0.8,
               colors_bgr[current_mask], 2, cv.LINE_AA)
    cv.putText(frame, f'Color Dominante: {current_dominant}',
               (10, 400), cv.FONT_HERSHEY_SIMPLEX, 0.8,
               colors_bgr[current_dominant], 2, cv.LINE_AA)

    # Mostrar ventanas
    cv.imshow('Camara', frame)
    cv.imshow('Mascara', mask)
    cv.imshow('Color Detectado', res)

    # Contar pixeles detectados
    red_count = cv.countNonZero(red_mask)
    blue_count = cv.countNonZero(blue_mask)
    green_count = cv.countNonZero(green_mask)
    yellow_count = cv.countNonZero(yellow_mask)
    orange_count = cv.countNonZero(orange_mask)
    purple_count = cv.countNonZero(purple_mask)
    pink_count = cv.countNonZero(pink_mask)
    brown_count = cv.countNonZero(brown_mask)

    colors_count = {
        'Rojo': red_count,
        'Azul': blue_count,
        'Verde': green_count,
        "Amarillo": yellow_count,
        "Naranja": orange_count,
        "Morado": purple_count,
        "Rosa": pink_count,
        "Cafe": brown_count
    }

    dominant_color = max(colors_count, key=colors_count.get)
    if dominant_color != current_dominant:
        current_dominant = dominant_color
        print(f'Color Dominante: {current_dominant}')

    # Leer teclas para cambiar máscara
    key = cv.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('1'):
        current_mask = 'Rojo'
    elif key == ord('2'):
        current_mask = 'Verde'
    elif key == ord('3'):
        current_mask = 'Azul'
    elif key == ord('4'):
        current_mask = 'Amarillo'
    elif key == ord('5'):
        current_mask = 'Naranja'
    elif key == ord('6'):
        current_mask = 'Morado'
    elif key == ord('7'):
        current_mask = 'Rosa'
    elif key == ord('8'):
        current_mask = 'Cafe'
cap.release()
cv.destroyAllWindows()
