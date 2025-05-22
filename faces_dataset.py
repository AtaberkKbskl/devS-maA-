import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class FacesDataset(Dataset):
    """
    Yüz resimlerinin otomatik kodlayıcı eğitimi için yüklenmesi.
    klasördeki tüm .jpg, .jpeg, .png dosyalarını OKUYUP,
    64×64 boyutuna indirger ve Tensor formatına çevirir.
    """
    def __init__(self, folder_path: str):
        # Tüm resim dosyası yollarını al
        self.paths = [os.path.join(folder_path, f)
                      for f in os.listdir(folder_path)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        # Dönüştürücüler: PILImage -> Resize -> Tensor
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()  # [0,1] aralığında float32 tensor
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        # Görüntüyü BGR olarak oku, RGB'ye çevir, transform uygula
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)
