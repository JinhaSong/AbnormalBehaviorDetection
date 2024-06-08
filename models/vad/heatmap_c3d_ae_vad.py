import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from tqdm import tqdm


class HeatmapC3D_AE_VAD(pl.LightningModule):
    def __init__(self, heatmap_c3d, autoencoder, lr=1e-3, wd=1e-5, max_frames=128):
        super(HeatmapC3D_AE_VAD, self).__init__()
        self.heatmap_c3d = heatmap_c3d
        self.autoencoder = autoencoder
        self.lr = lr
        self.wd = wd
        self.max_frames = max_frames
        self.val_step_outputs = []
        self.auroc = torchmetrics.classification.BinaryAUROC()
        self.train_epoch_losses = []

    def forward(self, x):
        features = self.heatmap_c3d(x)
        reconstructed = self.autoencoder(features)
        return reconstructed

    def training_step(self, batch, batch_idx):
        x = batch
        features = self.heatmap_c3d(x)
        reconstructed = self.autoencoder(features)
        loss = nn.functional.mse_loss(reconstructed, features)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['video_data']
        label = batch['label']

        features = self.heatmap_c3d(x)
        reconstructed = self.autoencoder(features)
        loss = nn.functional.mse_loss(reconstructed, features)
        self.log('val_loss', loss, prog_bar=True, batch_size=x.size(0))

        error = nn.functional.mse_loss(reconstructed, features, reduction='none')

        if error.dim() == 5:
            y_preds = error.mean(dim=[1, 2, 3, 4])
        elif error.dim() == 4:
            y_preds = error.mean(dim=[1, 2, 3])
        elif error.dim() == 3:
            y_preds = error.mean(dim=[1, 2])
        elif error.dim() == 2:
            y_preds = error.mean(dim=1)
        else:
            raise ValueError(f"Unexpected tensor dimensions: {error.dim()}")

        y_trues = label.float()

        self.val_step_outputs.append((y_preds, y_trues))
        return loss

    def on_validation_epoch_end(self):
        y_preds = torch.cat([x[0] for x in self.val_step_outputs], dim=0)
        y_trues = torch.cat([x[1] for x in self.val_step_outputs], dim=0)

        if len(torch.unique(y_trues)) > 1:
            auc = self.auroc(y_preds.cpu(), y_trues.cpu())
            self.log('val_auc', auc, prog_bar=True)
            tqdm.write(
                f'Epoch {self.current_epoch}: val_loss: {self.trainer.callback_metrics["val_loss"]:.4f}, val_auc: {auc:.4f}')
        else:
            tqdm.write('Only one class present in y_true. ROC AUC score is not defined.')
        self.val_step_outputs.clear()

        if 'train_loss' in self.trainer.callback_metrics:
            self.train_epoch_losses.append(self.trainer.callback_metrics['train_loss'].cpu().item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint['epoch'] = self.current_epoch
        checkpoint['global_step'] = self.global_step
        checkpoint['train_epoch_losses'] = self.train_epoch_losses

    def on_load_checkpoint(self, checkpoint):
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_epoch_losses = checkpoint.get('train_epoch_losses', [])