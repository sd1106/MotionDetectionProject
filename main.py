import cv2
import numpy as np
import pyautogui as pyauto

VideoCapture = cv2.VideoCapture(0)
pyauto.FAILSAFE = False

bgSubtractor = cv2.createBackgroundSubtractorMOG2()
kernal = np.ones((5, 5), np.uint8)

cv2.namedWindow('Final Frame', cv2.WINDOW_NORMAL)

onlyOnce = True

while True:

    ret, frame = VideoCapture.read()
    if not ret:
        break

    #cv2.imshow('Started Recording', frame)

    mask = bgSubtractor.apply(frame)
    #cv2.imshow('Remove Background', mask)

    erode = cv2.erode(mask, kernal, iterations=1)
    #cv2.imshow('Erode', erode)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Converted to Gray', gray)

    motion = cv2.bitwise_and(gray, gray, mask=erode)
    #cv2.imshow('Bitwise Motion', motion)

    _, thresh = cv2.threshold(motion, 223, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Thresholded', thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        largest = contours[0]
        area = cv2.contourArea(largest)

        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        xCenter = x + w / 2
        yCenter = y + h / 2

        width, height  = pyauto.size()
        frame_h, frame_w = frame.shape[:2]

        mouse_x = int(xCenter * width / frame_w)
        mouse_y = int(yCenter * height / frame_h)

        pyauto.moveTo(mouse_x, mouse_y)

        cv2.imshow('Final Frame', frame)


    if (onlyOnce):
        print("Frame: " + str(type(frame)))
        print("Mask: " + str(type(mask)))
        print("Erode: " + str(type(erode)))
        print("Gray: " + str(type(gray)))
        print("Motion: " + str(type(motion)))
        print("Threshold: " + str(type(thresh)))
        print("Final: " + str(type(frame)))
        onlyOnce = False


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break