import cv2
import os
from gan_anonymizer import GanAnonymizer
from performance_monitor import PerformanceMonitor
from foto_face_detector import PhotoFaceDetector
from video_face_detector import VideoFaceDetector

def run_image_mode(image_path: str, gan_model: GanAnonymizer):
    """
    Tek bir fotoğraf üzerinde:
     - PhotoFaceDetector ile yüz tespiti, landmark çizimi
     - GAN ile anonimleştirme
     - Sonuçları ekrana ve disk'e kaydetme
    """
    print("[INFO] Fotoğraf modu başlatılıyor...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Görüntü yüklenemedi: {image_path}")
        return

    detector = PhotoFaceDetector(anonymizer=gan_model, similarity_threshold=50)
    ano_img, det_img = detector.process_image(img)

    os.makedirs("outputs", exist_ok=True)
    path_ano = "outputs/anonymized_result.jpg"
    path_det = "outputs/detected_with_landmarks.jpg"
    cv2.imwrite(path_ano, ano_img)
    cv2.imwrite(path_det, det_img)
    print(f"[INFO] Kaydedildi:\n - {path_ano}\n - {path_det}")

    cv2.imshow("Anonymized", ano_img)
    cv2.imshow("Detections", det_img)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q veya ESC
            break
    cv2.destroyAllWindows()

def run_video_mode(source, gan_model: GanAnonymizer):
    """
    Gerçek zamanlı video/webcam:
     - VideoFaceDetector ile her karede tespit & anonimleştirme
     - FPS gösterimi
    """
    print("[INFO] Video modu başlatılıyor...")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Video açılamadı:", source)
        return

    perf = PerformanceMonitor()
    detector = VideoFaceDetector(anonymizer=gan_model, similarity_threshold=50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video sona erdi.")
            break

        perf.update()
        det_img, ano_img = detector.process_frame(frame)

        # FPS'i anonimlenmiş görüntüye yaz
        fps = perf.get_fps()
        cv2.putText(ano_img, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Detections & Landmarks", det_img)
        cv2.imshow("Anonymized Faces", ano_img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("=== YÜZ ANONİMLEŞTİRME SİSTEMİ ===")
    print("[INFO] GAN modeli yükleniyor...")
    gan = GanAnonymizer("saved_models/generator2.pth")

    choice = input("Fotoğraf için 'f', video için 'v': ").strip().lower()
    if choice == 'f':
        path = input("Fotoğraf dosya yolu: ").strip()
        run_image_mode(path, gan)
    elif choice == 'v':
        src = input("Video kaynağı (0=webcam veya dosya): ").strip()
        try:
            source = int(src)
        except ValueError:
            source = src
        run_video_mode(source, gan)
    else:
        print("[ERROR] Geçersiz seçim. Lütfen 'f' veya 'v' girin.")

if __name__ == "__main__":
    main()
