import torch
import torch.nn as nn

class ImprovedAutoEncoder(nn.Module):
    def __init__(self):
        super(ImprovedAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),     # -> [64, 64, 64]
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),   # -> [128, 32, 32]
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),  # -> [256, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),  # -> [512, 8, 8]
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # -> [256, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # -> [128, 32, 32]
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # -> [64, 64, 64]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # -> [3, 128, 128]
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
