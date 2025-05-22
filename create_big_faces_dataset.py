import os
import cv2
import numpy as np

# faces klasörünü yeniden oluştur (varsa sil, temiz başla)
import shutil
shutil.rmtree('faces', ignore_errors=True)
os.makedirs('faces', exist_ok=True)

# 200 adet yapay yüz üretelim
for i in range(200):
    img = np.ones((64, 64, 3), dtype=np.uint8) * 255

    # Rasgele yüz özellikleri
    center = (np.random.randint(20, 44), np.random.randint(20, 44))
    axes = (np.random.randint(10, 20), np.random.randint(10, 20))
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    cv2.ellipse(img, center, axes, 0, 0, 360, color, -1)

    eye_color = tuple(np.random.randint(0, 255, size=3).tolist())
    cv2.circle(img, (center[0] - 8, center[1] - 8), 3, eye_color, -1)
    cv2.circle(img, (center[0] + 8, center[1] - 8), 3, eye_color, -1)

    mouth_color = tuple(np.random.randint(0, 255, size=3).tolist())
    cv2.ellipse(img, (center[0], center[1] + 10), (10, 5), 0, 0, 180, mouth_color, 2)

    cv2.imwrite(f"faces/face_{i}.png", img)

print("[INFO] 200 adet yüz görüntüsü üretildi ve 'faces/' klasörüne kaydedildi.")
