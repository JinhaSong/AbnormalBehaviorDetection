import os
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from tqdm import tqdm


class HeatmapC3D_MemAE_VAD(pl.LightningModule):
    def __init__(self, heatmap_c3d, memautoencoder, lr=1e-3, wd=1e-5, max_frames=128, save_dir="checkpoints", config=None):
        super(HeatmapC3D_MemAE_VAD, self).__init__()
        self.heatmap_c3d = heatmap_c3d
        self.memautoencoder = memautoencoder
        self.lr = lr
        self.wd = wd
        self.max_frames = max_frames
        self.val_step_outputs = []
        self.train_step_outputs = []
        self.auroc = torchmetrics.classification.BinaryAUROC()
        self.train_epoch_losses = []
        self.best_val_auc = -float('inf')
        self.best_epoch = -1
        self.save_dir = save_dir
        self.config = config if config else {}
        self.chunk_size = 16
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, x):
        input_chunks = torch.split(x, self.chunk_size, dim=2)

        features_list = []
        for chunk in input_chunks:
            chunk_features = self.heatmap_c3d(chunk)
            features_list.append(chunk_features)

        features = torch.cat(features_list, dim=2)  # Concatenating along the temporal dimension
        features = features.permute(0, 2, 1, 3)
        return self.memautoencoder(features)

    def training_step(self, batch, batch_idx):
        x, label = batch

        # Forward pass
        out = self(x)
        reconstructed = out["recon"]

        input_chunks = torch.split(x, self.chunk_size, dim=2)
        features_list = []
        for chunk in input_chunks:
            chunk_features = self.heatmap_c3d(chunk)
            features_list.append(chunk_features)

        features = torch.cat(features_list, dim=2)  # Concatenating along the temporal dimension
        features = features.permute(0, 2, 1, 3)

        # Calculate losses
        loss_recon = nn.functional.mse_loss(reconstructed, features)

        loss_sparsity = (
            torch.mean(torch.sum(-out["att_weight3"] * torch.log(out["att_weight3"] + 1e-12), dim=1))
            + torch.mean(torch.sum(-out["att_weight2"] * torch.log(out["att_weight2"] + 1e-12), dim=1))
            + torch.mean(torch.sum(-out["att_weight1"] * torch.log(out["att_weight1"] + 1e-12), dim=1))
        )

        loss_all = self.config.get("lam_recon", 1.0) * loss_recon + self.config.get("lam_sparse", 1.0) * loss_sparsity

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

        self.train_step_outputs.append((y_preds, label.float()))

        self.log('train_loss', loss_all, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=x.size(0))
        return loss_all

    def on_train_epoch_end(self):
        y_preds = torch.cat([x[0] for x in self.train_step_outputs], dim=0)
        y_trues = torch.cat([x[1] for x in self.train_step_outputs], dim=0)

        if len(torch.unique(y_trues)) > 1:
            auc = self.auroc(y_preds.cpu(), y_trues.cpu())
            self.log('train_auc', auc, prog_bar=True)
            tqdm.write(f'Epoch {self.current_epoch}: train_loss: {self.trainer.callback_metrics["train_loss"]:.4f}, train_auc: {auc:.4f}')
        else:
            tqdm.write('Only one class present in y_true. ROC AUC score is not defined.')
        self.train_step_outputs.clear()

        self.save_model(f'epoch_{self.current_epoch}_train')

    def validation_step(self, batch, batch_idx):
        x, label = batch

        # Forward pass
        out = self(x)
        reconstructed = out["recon"]

        input_chunks = torch.split(x, self.chunk_size, dim=2)
        features_list = []
        for chunk in input_chunks:
            chunk_features = self.heatmap_c3d(chunk)
            features_list.append(chunk_features)

        features = torch.cat(features_list, dim=2)  # Concatenating along the temporal dimension
        features = features.permute(0, 2, 1, 3)

        # Calculate losses
        loss_recon = nn.functional.mse_loss(reconstructed, features)
        loss_sparsity = (
            torch.mean(torch.sum(-out["att_weight3"] * torch.log(out["att_weight3"] + 1e-12), dim=1))
            + torch.mean(torch.sum(-out["att_weight2"] * torch.log(out["att_weight2"] + 1e-12), dim=1))
            + torch.mean(torch.sum(-out["att_weight1"] * torch.log(out["att_weight1"] + 1e-12), dim=1))
        )

        loss_all = self.config.get("lam_recon", 1.0) * loss_recon + self.config.get("lam_sparse", 1.0) * loss_sparsity
        self.log('val_loss', loss_all, prog_bar=True, batch_size=x.size(0))

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
        return loss_all

    def on_validation_epoch_end(self):
        y_preds = torch.cat([x[0] for x in self.val_step_outputs], dim=0)
        y_trues = torch.cat([x[1] for x in self.val_step_outputs], dim=0)

        if len(torch.unique(y_trues)) > 1:
            auc = self.auroc(y_preds.cpu(), y_trues.cpu())
            self.log('val_auc', auc, prog_bar=True)
            tqdm.write(f'Epoch {self.current_epoch}: val_loss: {self.trainer.callback_metrics["val_loss"]:.4f}, val_auc: {auc:.4f}')

            if auc > self.best_val_auc:
                self.best_val_auc = auc
                self.best_epoch = self.current_epoch
                self.save_model(f'best_epoch_{self.best_epoch}')
        else:
            tqdm.write('Only one class present in y_true. ROC AUC score is not defined.')
        self.val_step_outputs.clear()

        if 'train_loss' in self.trainer.callback_metrics:
            self.train_epoch_losses.append(self.trainer.callback_metrics['train_loss'].cpu().item())

        self.save_model(f'epoch_{self.current_epoch}_val')

    def save_model(self, epoch_label):
        model_path = os.path.join(self.save_dir, f'{epoch_label}.pt')
        torch.save(self.state_dict(), model_path)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint['epoch'] = self.current_epoch
        checkpoint['global_step'] = self.global_step
        checkpoint['train_epoch_losses'] = self.train_epoch_losses
        checkpoint['best_val_auc'] = self.best_val_auc
        checkpoint['best_epoch'] = self.best_epoch

    def on_load_checkpoint(self, checkpoint):
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_epoch_losses = checkpoint.get('train_epoch_losses', [])
        self.best_val_auc = checkpoint.get('best_val_auc', -float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', -1)
