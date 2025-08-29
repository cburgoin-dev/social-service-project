import cv2 as cv
import numpy as np

def show_menu(frame):
    """Dibuja instrucciones en pantalla."""
    menu = [
        "1. Rojo",
        "2. Verde",
        "3. Azul",
        "4. Amarillo",
        "5. Naranja",
        "6. Morado",
        "7. Rosa",
        "8. Café",
        "ESC - Salir"
    ]
    y0, dy = 30, 30  # posición inicial y separación
    for i, line in enumerate(menu):
        y = y0 + i*dy
        cv.putText(frame, line, (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (255, 255, 255), 2, cv.LINE_AA)

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

cap = cv.VideoCapture(0)

current_mask = "Red"  # máscara inicial
current_dominant = None

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
    if current_mask == 'Red':
        mask = red_mask
    elif current_mask == 'Green':
        mask = green_mask
    elif current_mask == 'Blue':
        mask = blue_mask
    elif current_mask == 'Yellow':
        mask = yellow_mask
    elif current_mask == 'Orange':
        mask = orange_mask
    elif current_mask == 'Purple':
        mask = purple_mask
    elif current_mask == 'Pink':
        mask = pink_mask
    elif current_mask == 'Brown':
        mask = brown_mask   
    res = cv.bitwise_and(frame, frame, mask=mask)

    # --- Mostrar menú en pantalla ---
    show_menu(frame)
    cv.putText(frame, f'Current Mask: {current_mask}',
               (10, 160), cv.FONT_HERSHEY_SIMPLEX, 0.8,
               (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, f'Dominant Color: {current_dominant}',
               (10, 200), cv.FONT_HERSHEY_SIMPLEX, 0.8,
               (0, 255, 255), 2, cv.LINE_AA)

    # Mostrar ventanas
    cv.imshow('Original + Menu', frame)
    cv.imshow('Mask', mask)
    cv.imshow('Detected Color', res)

    # Contar pixeles detectados
    red_count = cv.countNonZero(red_mask)
    blue_count = cv.countNonZero(blue_mask)
    green_count = cv.countNonZero(green_mask)
    yellow_count = cv.countNonZero(yellow_mask)
    orange_count = cv.countNonZero(orange_mask)
    purple_count = cv.countNonZero(purple_mask)
    pink_count = cv.countNonZero(pink_mask)
    brown_count = cv.countNonZero(brown_mask)
    
    colors = {
        'Red': red_count,
        'Blue': blue_count,
        'Green': green_count,
        "Yellow": yellow_count,
        "Orange": orange_count,
        "Purple": purple_count,
        "Pink": pink_count,
        "Brown": brown_count
    }

    dominant_color = max(colors, key=colors.get)
    if dominant_color != current_dominant:
        current_dominant = dominant_color
        print(f'Dominant Color: {current_dominant}')

    # Leer teclas para cambiar máscara
    key = cv.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('1'):
        current_mask = 'Red'
    elif key == ord('2'):
        current_mask = 'Green'
    elif key == ord('3'):
        current_mask = 'Blue'
    elif key == ord('4'):
        current_mask = 'Yellow'
    elif key == ord('5'):
        current_mask = 'Orange'
    elif key == ord('6'):
        current_mask = 'Purple'
    elif key == ord('7'):
        current_mask = 'Pink'
    elif key == ord('8'):
        current_mask = 'Brown'
cap.release()
cv.destroyAllWindows()
