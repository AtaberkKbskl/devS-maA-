import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from scipy.spatial import procrustes

class VideoFaceDetector:
    def __init__(self, anonymizer=None, similarity_threshold=50, max_faces=5):
        self.anonymizer = anonymizer
        self.similarity_threshold = similarity_threshold

        # YOLOv8 - insan (person) tespiti
        self.yolo_human = YOLO('yolov8n.pt')

        # YOLOv8-face - yüz tespiti
        self.yolo_face = YOLO('yolov8n-face.pt')

        # Haar cascade (isteğe bağlı)
        haar = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar)

        # MediaPipe FaceMesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_similarity(self, pts, bbox):
        if len(pts) < 150:
            return 0.0
        ideal = np.linspace([0, 0], [1, 1], 150)
        det = np.array(pts)[:150]
        _, _, disp = procrustes(ideal, det)
        return max(0, min(100, (1 - disp) * 100))

    def draw_landmarks(self, img, face_landmarks):
        h, w, _ = img.shape
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        det_img = frame.copy()
        ano_img = frame.copy()

        # 1) YOLOv8 Human tespiti
        for res in self.yolo_human(frame):
            for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
                if res.names[int(cls)].lower() == 'person':
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(det_img, "HUMAN", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2) YOLOv8 Face tespiti + MediaPipe + GAN anonimleştirme
        for res in self.yolo_face(frame):
            for box in res.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                # 2.a) sarı kutu ile yüz deteksiyonu
                cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(det_img, "FACE", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 2.b) ROI olarak crop al
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue

                # 2.c) Bu crop'u MediaPipe ile işle
                rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                mesh_res = self.face_mesh.process(rgb_roi)

                # 2.d) Landmark'ları çiz (isteğe bağlı)
                if mesh_res.multi_face_landmarks:
                    for lm_set in mesh_res.multi_face_landmarks:
                        for lm in lm_set.landmark:
                            gx = int(lm.x * (x2 - x1)) + x1
                            gy = int(lm.y * (y2 - y1)) + y1
                            cv2.circle(det_img, (gx, gy), 1, (0, 255, 0), -1)

                # 2.e) GAN anonimleştirme
                if self.anonymizer:
                    anon = self.anonymizer.anonymize_face(face_roi)
                    if anon is not None:
                        anon = cv2.resize(anon, (x2 - x1, y2 - y1))
                        ano_img[y1:y2, x1:x2] = anon
                        cv2.rectangle(ano_img, (x1, y1), (x2, y2),
                                      (0, 0, 255), 2)

        return det_img, ano_img

    def process_video(self, source=0, performance_monitor=None):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError("Video açılamadı")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if performance_monitor:
                performance_monitor.update()

            det, ano = self.process_frame(frame)

            if performance_monitor:
                fps = performance_monitor.get_fps()
                cv2.putText(ano, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Detections & Landmarks", det)
            cv2.imshow("Anonymized Faces", ano)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        det, ano = self.process_frame(img)
        cv2.imshow("Detections & Landmarks", det)
        cv2.imshow("Anonymized Faces", ano)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return det, ano
