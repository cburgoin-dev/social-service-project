import cv2 as cv
import numpy as np

# def rescaleFrame(frame, scale=0.75):
#     # Images, Videos and Live Video
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)

#     dimensions = (width, height)

#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# img = cv.imread('Photos/dog.jpg')
# resized_img = rescaleFrame(img, 0.1)

# hsv = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)

# lower_red1 = np.array([0, 120, 70])
# upper_red1 = np.array([10, 255, 255])

# lower_red2 = np.array([160, 120, 70])
# upper_red2 = np.array([180, 255, 255])

# # lower_blue = np.array([100, 150, 50])
# # upper_blue = np.array([140, 255, 255])

# mask1 = cv.inRange(hsv, lower_red1, upper_red1)
# mask2 = cv.inRange(hsv, lower_red2, upper_red2)

# mask = mask1 + mask2

# res = cv.bitwise_and(resized_img,resized_img, mask=mask)

# cv.imshow('Original', resized_img)
# cv.imshow('Mask', mask)
# cv.imshow('Detected Color', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 255, 255])

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    red_mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(red_mask1, red_mask2)

    green_mask = cv.inRange(hsv, lower_green, upper_green)
    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)

    res = cv.bitwise_and(frame, frame, mask=red_mask)

    cv.imshow('Original', frame)
    cv.imshow('Mask', red_mask)
    cv.imshow('Detected Color', res)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()