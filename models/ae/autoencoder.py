import torch
import torch.nn as nn
import pytorch_lightning as pl

class AutoEncoder(pl.LightningModule):
    def __init__(self, input_dim=512, latent_dim=128, lr=1e-3):
        super(AutoEncoder, self).__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.ReLU(),
        )
        self.lr = lr

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x = batch
        reconstructed = self(x)
        loss = nn.functional.mse_loss(reconstructed, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        reconstructed = self(x)
        loss = nn.functional.mse_loss(reconstructed, x)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
