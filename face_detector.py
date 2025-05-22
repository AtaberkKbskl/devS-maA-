import cv2
import mediapipe as mp
import numpy as np
import os
from ultralytics import YOLO
from scipy.spatial import procrustes

class FaceDetector:
    def __init__(self, anonymizer=None, similarity_threshold=50, max_faces=5):
        self.anonymizer = anonymizer
        self.similarity_threshold = similarity_threshold

        # Haar Cascade yükle
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(haar_path):
            raise FileNotFoundError(f"Haarcascade dosyası bulunamadı: {haar_path}")
        self.face_cascade = cv2.CascadeClassifier(haar_path)

        # MediaPipe FaceMesh (468 landmarks)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # YOLOv8 model
        self.yolo_model = YOLO("yolov8n.pt")

    def detect_faces_with_yolo(self, frame):
        results = self.yolo_model(frame)
        faces = []
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                label = result.names[int(cls)]
                if "face" in label.lower():
                    x1, y1, x2, y2 = map(int, box)
                    faces.append((x1, y1, x2, y2))
        return faces

    def calculate_similarity(self, landmarks, bbox):
        if not landmarks or len(landmarks) < 150:
            return 0.0
        ideal = np.linspace([0, 0], [1, 1], 150)
        detected = np.array(landmarks)[:150]
        _, _, disparity = procrustes(ideal, detected)
        score = max(0, min(100, (1 - disparity) * 100))

        x1, y1, x2, y2 = bbox
        min_x, min_y = detected[:, 0].min(), detected[:, 1].min()
        max_x, max_y = detected[:, 0].max(), detected[:, 1].max()
        if not (x1 <= min_x <= x2 and y1 <= min_y <= y2 and x1 <= max_x <= x2 and y1 <= max_y <= y2):
            score *= 0.5
        return score

    def draw_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    def process_image(self, image):
        h_img, w_img = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Haar Cascade çizimi
        for (x, y, w, h) in self.face_cascade.detectMultiScale(gray, 1.1, 4):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 1)

        # YOLO ile yüz tespiti
        yolo_faces = self.detect_faces_with_yolo(image)
        # MediaPipe ile landmark tespiti
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                points = [(int(l.x * w_img), int(l.y * h_img)) for l in face_landmarks.landmark]
                # Kutu belirleme
                if i < len(yolo_faces):
                    x1, y1, x2, y2 = yolo_faces[i]
                else:
                    x1, y1, x2, y2 = 0, 0, w_img, h_img
                # Genişletme
                mw = int((x2 - x1) * 0.2)
                mh = int((y2 - y1) * 0.3)
                x1, y1 = max(0, x1 - mw), max(0, y1 - mh)
                x2, y2 = min(w_img, x2 + mw), min(h_img, y2 + mh)

                sim = self.calculate_similarity(points, (x1, y1, x2, y2))
                if self.anonymizer and sim > self.similarity_threshold:
                    region = image[y1:y2, x1:x2]
                    anon = self.anonymizer.anonymize_face(region)
                    if anon is not None:
                        image[y1:y2, x1:x2] = anon

                # Çizimler
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self.draw_landmarks(image, face_landmarks)

        cv2.imshow("Fotoğraf Anonimleştirme", image)

    def process_video(self, video_source=0, performance_monitor=None):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("[ERROR] Video açılamadı!")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if performance_monitor:
                performance_monitor.update()

            self.process_image(frame)

            if performance_monitor:
                fps = performance_monitor.get_fps()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
