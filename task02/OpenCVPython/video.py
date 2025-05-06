import cv2
import mediapipe as mp
from ultralytics import YOLO
# haarcascade_frontalface_default.xml
# haarcascade_fullbody.xml
# haarcascade_frontalcatface.xml
# haarcascade_lowerbody.xml
# haarcascade_upperbody.xml
# haarcascade_lowerbody.xml
face_cascades = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")


# img = cv2.imread('D:\\ANACONDA\\Artificial intelligence\\task02\\OpenCVPython\\01.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = face_cascades.detectMultiScale(img_gray)
# # print(face)
# for (x ,y ,w, h) in faces:
#     cv2.rectangle(img,(x, y),(x + w, y + h),(0, 255, 0),2)

# cv2.imshow('Result', img)

# cv2.waitKey(0)

# v1------------------------------------------------------------------------------------------------------

# cap = cv2.VideoCapture("D:\\ANACONDA\\Artificial intelligence\\task02\\OpenCVPython\\02.mp4")
# cap2 = cv2.VideoCapture("D:\\ANACONDA\\Artificial intelligence\\task02\\OpenCVPython\\03.mp4")

# # print(cap)

# while True:
#     success, frame = cap2.read()
#     # cv2.imshow("camera", frame)
#     img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow('Prewie', frame)
#     faces = face_cascades.detectMultiScale(img_gray)

#     for (x ,y ,w, h) in faces:
#         cv2.rectangle(frame,(x, y),(x + w, y + h),(0, 255, 0),2)
#     cv2.imshow('Result', frame)

#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break


# v2------------------------------------------------------------------------------------------------------
# MediaPipe (от Google)

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_draw = mp.solutions.drawing_utils

# cap = cv2.VideoCapture("D:\\ANACONDA\\Artificial intelligence\\task02\\OpenCVPython\\02.mp4")
# cap2 = cv2.VideoCapture("D:\\ANACONDA\\Artificial intelligence\\task02\\OpenCVPython\\03.mp4")
# cap3 = cv2.VideoCapture("D:\\ANACONDA\\Artificial intelligence\\task02\\OpenCVPython\\bf3.avi")

# while True:
#     success, img = cap.read()
#     if not success:
#         break
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = pose.process(img_rgb)

#     if results.pose_landmarks:
#         mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#     cv2.imshow("Pose Detection", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break




# v3------------------------------------------------------------------------------------------------------
# YOLO (You Only Look Once)

# Загрузим предобученную модель YOLOv8
model = YOLO('yolov8n.pt')  # или yolov8s.pt / yolov8m.pt для большей точности

cap = cv2.VideoCapture("D:\\ANACONDA\\Artificial intelligence\\task02\\OpenCVPython\\02.mp4")
cap2 = cv2.VideoCapture("D:\\ANACONDA\\Artificial intelligence\\task02\\OpenCVPython\\03.mp4")
cap3 = cv2.VideoCapture("D:\\ANACONDA\\Artificial intelligence\\task02\\OpenCVPython\\bf3.avi")

while True:
    success, frame = cap3.read()
    if not success:
        break

    # Обнаружение объектов
    results = model(frame)[0]

    # Отрисовка результатов
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if results.names[cls_id] == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


