import cv2
import time
import mediapipe as mp
import numpy as np


class FaceDetector:
    def __init__(self, min_detection_con=0.5):
        self.min_detection_con = 0.5
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_con)

    def find_faces(self, img, draw=True, score=False):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(rgb_img)
        bboxes = []
        if self.results.detections:
            for id , detection in enumerate(self.results.detections):
                # print(id, detection)
                # mp_draw.draw_detection(img, detection)
                # print(detection.score)

                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                # bbox = np.array([bbox_c.xmin * iw, bbox_c.ymin * ih, bbox_c.width * iw, bbox_c.height * iw])
                bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                       int(bbox_c.width * iw), int(bbox_c.height * ih)
                if score:
                    bboxes.append([bbox, detection.score])
                else:
                    bboxes.append(bbox)

                if draw:
                    self.fancy_boxes(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 50, 255), 2)

        return img, bboxes

    def fancy_boxes(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        # top left x, y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # top right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # bottom left x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # bottom right x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

def main():
    capture = cv2.VideoCapture(0)
    p_time = 0
    face_detector = FaceDetector()
    while True:
        _, img = capture.read()
        img, bboxes = face_detector.find_faces(img)

        print(bboxes)
        c_time = time.time()
        fps = int(1 / (c_time - p_time))
        p_time = c_time
        cv2.putText(img, str(fps), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow('live', img)
        cv2.waitKey(10)


if __name__ == '__main__':
    main()