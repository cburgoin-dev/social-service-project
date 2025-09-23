import cv2 
import numpy as np
from ultralytics import YOLO


model = YOLO("yolov8n.pt")



results = model("fruits.jpg")[0]
high_conf_boxes = [box for box in results.boxes if float(box.conf[0])>=0.5]
results.boxes = high_conf_boxes

img = results.plot() 
cv2.imshow("Resultado", img)


for box in results.boxes:
    class_id = int(box.cls[0])
    class_name = results.names[class_id]
    confidence = box.conf[0]
    print(f"Objeto: {class_name}, Confianza: {confidence:.2f}")
cv2.waitKey(0)
cv2.destroyAllWindows()


