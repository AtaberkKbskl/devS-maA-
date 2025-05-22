# test_gan_output_with_detection.py
import cv2
from gan_anonymizer import GanAnonymizer

# Görseli oku
image_path = "face.jpg"
img = cv2.imread(image_path)
if img is None:
    print("[ERROR] Görsel yüklenemedi.")
    exit()

# Haar cascade ile yüz tespiti
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

if len(faces) == 0:
    print("[ERROR] Yüz bulunamadı.")
    exit()

# İlk yüzü al
x, y, w, h = faces[0]
face_crop = img[y:y+h, x:x+w]

# 128x128'e yeniden boyutlandır (GAN girişi için)
face_crop_resized = cv2.resize(face_crop, (128, 128))

# GAN yükle
anonymizer = GanAnonymizer("saved_models/generator.pth")

# Anonimleştir
anonymized = anonymizer.anonymize_face(face_crop_resized)

# Kaydet
cv2.imwrite("outputs/original_face.jpg", face_crop_resized)
cv2.imwrite("outputs/anonymized_face.jpg", anonymized)

print("[INFO] Yüz başarıyla anonimleştirildi. outputs klasörünü kontrol et.")
