import cv2
import numpy as np
import time
from imutils.video import VideoStream
import mediapipe as mp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from face_detection.face_detection_module import FaceDetector

face_detector = FaceDetector()
capture = cv2.VideoCapture('./Pose estimation/videos/dance.mp4')
# capture = VideoStream(0).start()
mask_net = load_model('mask_detector.model')
p_time = 0

def utils(img, bboxes):
    faces = []
    for bbox in bboxes:
        start_x, start_y, end_x, end_y = bbox
        end_x += start_x
        end_y += start_y
        face = img[int(start_y):int(end_y), int(start_x): int(end_x)]
        face = cv2.resize(img, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        prediction = model([face])
        preds = np. argmax(prediction)
        if preds == 1:
            color = (0, 0, 255)
            text = 'no mask'
        else:
            color = (0, 255, 0)
            text = 'mask'
        cv2.rectangle(img, bbox, color, 1)
        cv2.putText(img, text, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    return img



def model(face):
    if len(face) > 0:
        face = np.array(face, dtype=np.float32)
        predictions = mask_net.predict(face, batch_size=32)
        return predictions



while True:
    _, img = capture.read()
    d_img, bboxes = face_detector.find_faces(img, draw=False, score=False)
    faces = utils(img, bboxes)

    c_time = time.time()
    fps = str(int(1/(c_time - p_time)))
    p_time = c_time

    cv2.putText(img, fps, (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.imshow('live', img)
    cv2.waitKey(20)


