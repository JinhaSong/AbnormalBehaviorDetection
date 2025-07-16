import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
import numpy as np


class HeatmapMTHPA_Contrastive(pl.LightningModule):
    def __init__(self, mthpa_model, lr=1e-4, wd=1e-5, save_dir="checkpoints", temperature=0.07):
        super().__init__()
        self.mthpa_model = mthpa_model
        self.lr = lr
        self.wd = wd
        self.save_dir = save_dir
        self.temperature = temperature
        self.train_features = None
        self.val_features = None
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, x):
        # x: (batch_size, time, objects, 224, 224)
        return self.mthpa_model(x)

    def info_nce_loss(self, anchor, pair, labels, temperature):
        # Flatten features to (B, D)
        anchor = anchor.view(anchor.size(0), -1)
        pair = pair.view(pair.size(0), -1)
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)
            pair = pair.unsqueeze(0)
            labels = labels.unsqueeze(0)
        anchor = F.normalize(anchor, dim=-1)
        pair = F.normalize(pair, dim=-1)
        sim = torch.sum(anchor * pair, dim=-1) / temperature  # (B,)
        loss_pos = -labels * torch.log(torch.sigmoid(sim) + 1e-12)
        loss_neg = -(1 - labels) * torch.log(1 - torch.sigmoid(sim) + 1e-12)
        loss = (loss_pos + loss_neg).mean()
        return loss

    def training_step(self, batch, batch_idx):
        anchor, pair, labels = batch  # (B, T, O, 224, 224), (B, T, O, 224, 224), (B,)
        anchor_feat = self.mthpa_model(anchor)
        pair_feat = self.mthpa_model(pair)
        loss = self.info_nce_loss(anchor_feat, pair_feat, labels, self.temperature)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=anchor.size(0), sync_dist=True)
        # Store anchor and pair features for visualization
        feats = torch.cat([anchor_feat.detach().cpu(), pair_feat.detach().cpu()], dim=0)  # (2B, D)
        labs = torch.cat([labels.detach().cpu(), labels.detach().cpu()], dim=0)  # (2B,)
        if self.train_features is None:
            self.train_features = (feats, labs)
        else:
            self.train_features = (
                torch.cat((self.train_features[0], feats), dim=0),
                torch.cat((self.train_features[1], labs), dim=0)
            )
        return loss

    def validation_step(self, batch, batch_idx):
        anchor, pair, labels = batch
        anchor_feat = self.mthpa_model(anchor)
        pair_feat = self.mthpa_model(pair)
        loss = self.info_nce_loss(anchor_feat, pair_feat, labels, self.temperature)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=anchor.size(0), sync_dist=True)
        # Store anchor and pair features for visualization
        feats = torch.cat([anchor_feat.detach().cpu(), pair_feat.detach().cpu()], dim=0)  # (2B, D)
        labs = torch.cat([labels.detach().cpu(), labels.detach().cpu()], dim=0)  # (2B,)
        if self.val_features is None:
            self.val_features = (feats, labs)
        else:
            self.val_features = (
                torch.cat((self.val_features[0], feats), dim=0),
                torch.cat((self.val_features[1], labs), dim=0)
            )
        return loss

    def on_train_epoch_end(self, unused=None):
        if self.train_features is not None:
            features, labels = self.train_features
            self.visualize_tsne(features, labels, "train", self.current_epoch)
            self.train_features = None
        self.save_model(self.current_epoch)

    def on_validation_epoch_end(self):
        if self.val_features is not None:
            features, labels = self.val_features
            self.visualize_tsne(features, labels, "val", self.current_epoch)
            self.val_features = None

    def on_train_start(self):
        self.log('val_loss', float('inf'), prog_bar=True, logger=True, sync_dist=True)

    def visualize_tsne(self, features, labels, stage, epoch):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import os

        features = features.numpy()
        labels = labels.numpy()
        features = features.reshape(features.shape[0], -1)
        if features.shape[1] < 2:
            print(f"[TSNE] Skipping TSNE visualization for {stage} at epoch {epoch} because feature dim < 2 (got {features.shape[1]})")
            return
        num_samples = features.shape[0]
        tsne_perplexity = min(30, max(1, (num_samples - 1) // 3))
        if num_samples <= tsne_perplexity:
            print(f"Skipping TSNE visualization for {stage} at epoch {epoch} due to insufficient samples.")
            return
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
        tsne_results = tsne.fit_transform(features)

        # Plot with color by label
        plt.figure(figsize=(10, 6))
        for label, color, name in zip([0, 1], ['green', 'red'], ['normal', 'abnormal']):
            idx = labels == label
            plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=color, label=name, alpha=0.7, s=20)
        plt.legend()
        plt.title(f't-SNE {stage} Epoch {epoch}')
        tsne_dir = os.path.join(self.save_dir, f'tsne_visualization')
        os.makedirs(tsne_dir, exist_ok=True)
        tsne_path = os.path.join(tsne_dir, f'tsne_{stage}_epoch_{epoch}.jpg')
        plt.savefig(tsne_path)
        plt.close()

        # wandb: log to a single panel per stage
        if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
            self.logger.experiment.log({f"tsne_{stage}": wandb.Image(tsne_path), "epoch": epoch})

    def visualize_temporal_saliency(self, saliency, class_name, epoch):
        plt.figure(figsize=(10, 4))
        plt.plot(saliency.cpu().numpy())
        plt.title(f'Temporal Saliency for {class_name} at Epoch {epoch}')
        plt.xlabel('Frame')
        plt.ylabel('Saliency')
        plt.grid(True)
        ts_dir = os.path.join(self.save_dir, f'temporal_saliency')
        os.makedirs(ts_dir, exist_ok=True)
        ts_path = os.path.join(ts_dir, f'{class_name}_temporal_saliency_epoch{epoch:03d}.png')
        plt.savefig(ts_path)
        plt.close()
        if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
            self.logger.experiment.log({f"{class_name}_temporal_saliency": wandb.Image(ts_path)})

    def save_model(self, epoch_label):
        model_path = os.path.join(self.save_dir, f'epoch{epoch_label:03d}.ckpt')
        self.trainer.save_checkpoint(model_path)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_state_dict'] = self.mthpa_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if 'model_state_dict' in checkpoint:
            self.mthpa_model.load_state_dict(checkpoint['model_state_dict'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        } 