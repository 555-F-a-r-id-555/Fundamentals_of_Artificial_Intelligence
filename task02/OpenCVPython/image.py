import cv2
import mediapipe as mp
# haarcascade_frontalface_default.xml
# haarcascade_fullbody.xml
# haarcascade_frontalcatface.xml
# haarcascade_lowerbody.xml
# haarcascade_upperbody.xml
# haarcascade_lowerbody.xml
face_cascades = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


img = cv2.imread('D:\\ANACONDA\\Artificial intelligence\\task02\\OpenCVPython\\01.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascades.detectMultiScale(img_gray)
# print(face)
for (x ,y ,w, h) in faces:
    cv2.rectangle(img,(x, y),(x + w, y + h),(0, 255, 0),2)

cv2.imshow('Result', img)

cv2.waitKey(0)
