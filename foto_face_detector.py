import cv2
import mediapipe as mp
import numpy as np
import os
from ultralytics import YOLO

class PhotoFaceDetector:
    def __init__(self, anonymizer=None, similarity_threshold=50):
        self.anonymizer = anonymizer
        self.similarity_threshold = similarity_threshold

        # Haar Cascade for face
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(haar_path):
            raise FileNotFoundError(f"Cascade not found: {haar_path}")
        self.face_cascade = cv2.CascadeClassifier(haar_path)

        # MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # YOLO for person detection
        self.yolo = YOLO("yolov8n.pt")

    def process_image(self, image_or_path):
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                print(f"[ERROR] Cannot load image: {image_or_path}")
                return None, None
        else:
            image = image_or_path.copy()

        h, w = image.shape[:2]
        det_img = image.copy()
        ano_img = image.copy()

        # 1. YOLO ile person tespiti
        results = self.yolo(image)
        for res in results:
            for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
                label = res.names[int(cls)]
                if label == "person":
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                    # Yeşil kutu + "human" etiketi
                    cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(det_img, "human", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    person_crop = image[y1:y2, x1:x2]

                    # 2. Bu kişi içinde yüz ara
                    gray_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray_crop, 1.1, 4)

                    for (fx, fy, fw, fh) in faces:
                        fx1, fy1 = x1 + fx, y1 + fy
                        fx2, fy2 = fx1 + fw, fy1 + fh

                        # Sarı kutu + "face" etiketi
                        cv2.rectangle(det_img, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                        cv2.putText(det_img, "face", (fx1, fy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Landmark çizimi (tam resme göre)
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        lm_res = self.face_mesh.process(rgb)
                        if lm_res.multi_face_landmarks:
                            for lm in lm_res.multi_face_landmarks[0].landmark:
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(det_img, (cx, cy), 1, (0, 255, 0), -1)

                        # GAN ile anonimleştir
                        face_crop = image[fy1:fy2, fx1:fx2]
                        if self.anonymizer:
                            anon = self.anonymizer.anonymize_face(face_crop)
                            if anon is not None:
                                resized_anon = cv2.resize(anon, (fw, fh))
                                ano_img[fy1:fy2, fx1:fx2] = resized_anon
                                cv2.rectangle(ano_img, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

        return ano_img, det_img
