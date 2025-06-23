import torch
import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self, input_dim):
        super(ConvAE, self).__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1))
        )

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(2)  # (batch_size, 1024) -> (batch_size, 1, 1, 1024)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.squeeze(2)  # (batch_size, 1, 1, 1024) -> (batch_size, 1, 1024)
        decoded = decoded.squeeze(1)  # (batch_size, 1, 1024) -> (batch_size, 1024)
        return decoded