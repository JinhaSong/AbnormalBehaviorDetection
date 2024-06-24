import os
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from tqdm import tqdm


class ResNet3D_Triplet(pl.LightningModule):
    def __init__(self, heatmap_c3d, lr=1e-3, wd=1e-5, save_dir="checkpoints"):
        super(ResNet3D_Triplet, self).__init__()
        self.heatmap_c3d = heatmap_c3d
        self.lr = lr
        self.wd = wd
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, x):
        return self.heatmap_c3d(x)

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor_features = self.heatmap_c3d(anchor)
        positive_features = self.heatmap_c3d(positive)
        negative_features = self.heatmap_c3d(negative)

        loss = self.triplet_loss(anchor_features, positive_features, negative_features)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=anchor.size(0))
        return loss

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        pos_dist = torch.nn.functional.mse_loss(anchor, positive, reduction='none')
        neg_dist = torch.nn.functional.mse_loss(anchor, negative, reduction='none')

        mask = (anchor != -1).float()
        pos_dist = (pos_dist * mask).sum() / mask.sum()
        neg_dist = (neg_dist * mask).sum() / mask.sum()

        loss = torch.relu(pos_dist - neg_dist + margin)
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor_features = self.heatmap_c3d(anchor)
        positive_features = self.heatmap_c3d(positive)
        negative_features = self.heatmap_c3d(negative)

        loss = self.triplet_loss(anchor_features, positive_features, negative_features)
        self.log('val_loss', loss, prog_bar=True, batch_size=anchor.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def save_model(self, epoch_label):
        model_path = os.path.join(self.save_dir, f'{epoch_label}.pt')
        torch.save(self.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['epoch'] = self.current_epoch
        checkpoint['global_step'] = self.global_step

    def on_load_checkpoint(self, checkpoint):
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']