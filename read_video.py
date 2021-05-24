import cv2
import numpy
import time

i = 0
start = time.time()

# cap = cv2.VideoCapture("rtsp://{user}:{pw}@{ip}:{port}".format(user, pw, ip, port))
cap = cv2.VideoCapture(0)
print('Initiated connection')
while True:
    ret, frame = cap.read()
    time.sleep(0.01)
    i += 1
    fps = i / (time.time() - start)
    new = frame.copy()
    new = cv2.putText(new, "FPS: {}".format(fps), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255))
    new = cv2.resize(new, (1920, 1080))
    cv2.imshow('Yeet', new)
    cv2.waitKey(1)