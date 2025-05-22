import cv2
import torch
import numpy as np
from autoencoder_model import ImprovedAutoEncoder

class GanAnonymizer:
    def __init__(self, weight_path="saved_models/generator1.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ImprovedAutoEncoder().to(self.device)

        # .pt veya .pth dosyası olabilir
        state = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        print(f"[INFO] GAN modeli yüklendi: {weight_path} (device={self.device})")

    def anonymize_face(self, face_region: np.ndarray) -> np.ndarray:
        if face_region is None or face_region.shape[0] < 10 or face_region.shape[1] < 10:
            print("[WARN] Geçersiz yüz bölgesi - atlanıyor")
            return None

        # 1. 128x128 ve RGB
        resized = cv2.resize(face_region, (128, 128))
        face_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        face_tensor = torch.from_numpy(face_rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

        # 2. GAN inference
        with torch.no_grad():
            output_tensor = self.model(face_tensor.to(self.device))

        out_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)

        # 3. BGR'ye çevir ve geri döndür
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
