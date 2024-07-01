import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb

class Heatmap3D_Triplet(pl.LightningModule):
    def __init__(self, heatmap_c3d, lr=1e-3, wd=1e-5, save_dir="checkpoints"):
        super(Heatmap3D_Triplet, self).__init__()
        self.heatmap_c3d = heatmap_c3d
        self.lr = lr
        self.wd = wd
        self.save_dir = save_dir
        self.train_features = None
        self.val_features = None
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, x):
        return self.heatmap_c3d(x)

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        pos_dist = torch.nn.functional.mse_loss(anchor, positive, reduction='none')
        neg_dist = torch.nn.functional.mse_loss(anchor, negative, reduction='none')

        mask = (anchor != -1).float()

        pos_dist = (pos_dist * mask).sum(dim=1) / mask.sum(dim=1)
        neg_dist = (neg_dist * mask).sum(dim=1) / mask.sum(dim=1)

        loss = torch.relu(pos_dist - neg_dist + margin)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        anchor, positive, negative, is_normal = batch
        anchor_features = self.heatmap_c3d(anchor)
        positive_features = self.heatmap_c3d(positive)
        negative_features = self.heatmap_c3d(negative)

        loss = self.triplet_loss(anchor_features, positive_features, negative_features)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=anchor.size(0),
                 sync_dist=True)

        if self.train_features is None:
            self.train_features = (
            anchor_features.detach().cpu(), positive_features.detach().cpu(), negative_features.detach().cpu(),
            is_normal.detach().cpu())
        else:
            self.train_features = (
                torch.cat((self.train_features[0], anchor_features.detach().cpu()), dim=0),
                torch.cat((self.train_features[1], positive_features.detach().cpu()), dim=0),
                torch.cat((self.train_features[2], negative_features.detach().cpu()), dim=0),
                torch.cat((self.train_features[3], is_normal.detach().cpu()), dim=0)
            )

        return loss

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative, is_normal = batch
        anchor_features = self.heatmap_c3d(anchor)
        positive_features = self.heatmap_c3d(positive)
        negative_features = self.heatmap_c3d(negative)

        loss = self.triplet_loss(anchor_features, positive_features, negative_features)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=anchor.size(0),
                 sync_dist=True)

        if self.val_features is None:
            self.val_features = (
            anchor_features.detach().cpu(), positive_features.detach().cpu(), negative_features.detach().cpu(),
            is_normal.detach().cpu())
        else:
            self.val_features = (
                torch.cat((self.val_features[0], anchor_features.detach().cpu()), dim=0),
                torch.cat((self.val_features[1], positive_features.detach().cpu()), dim=0),
                torch.cat((self.val_features[2], negative_features.detach().cpu()), dim=0),
                torch.cat((self.val_features[3], is_normal.detach().cpu()), dim=0)
            )

        return loss

    def on_train_epoch_end(self, unused=None):
        if self.train_features is not None:
            normal_mask = self.train_features[3].bool()
            abnormal_mask = ~normal_mask

            normal_features = (
                self.train_features[0][normal_mask],
                self.train_features[1][normal_mask],
                self.train_features[2][normal_mask]
            )
            abnormal_features = (
                self.train_features[0][abnormal_mask],
                self.train_features[1][abnormal_mask],
                self.train_features[2][abnormal_mask]
            )

            self.visualize_tsne(*normal_features, "train_normal", self.current_epoch)
            self.visualize_tsne(*abnormal_features, "train_abnormal", self.current_epoch)

            self.train_features = None

        self.save_model(self.current_epoch)

    def on_validation_epoch_end(self):
        if self.val_features is not None:
            normal_mask = self.val_features[3].bool()
            abnormal_mask = ~normal_mask

            normal_features = (
                self.val_features[0][normal_mask],
                self.val_features[1][normal_mask],
                self.val_features[2][normal_mask]
            )
            abnormal_features = (
                self.val_features[0][abnormal_mask],
                self.val_features[1][abnormal_mask],
                self.val_features[2][abnormal_mask]
            )

            self.visualize_tsne(*normal_features, "val_normal", self.current_epoch)
            self.visualize_tsne(*abnormal_features, "val_abnormal", self.current_epoch)

            self.val_features = None

    def on_train_start(self):
        self.log('val_loss', float('inf'), prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        lr_scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler_config]

    def visualize_tsne(self, anchor_features, positive_features, negative_features, stage, epoch):
        features = torch.cat((anchor_features, positive_features, negative_features), dim=0).numpy()
        num_samples = features.shape[0]
        tsne_perplexity = min(30, max(1, (num_samples - 1) // 3))

        if num_samples <= tsne_perplexity:
            print(f"Skipping TSNE visualization for {stage} at epoch {epoch} due to insufficient samples.")
            return

        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
        tsne_results = tsne.fit_transform(features)

        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_results[:anchor_features.size(0), 0], tsne_results[:anchor_features.size(0), 1], c='r', label='Anchor')
        plt.scatter(tsne_results[anchor_features.size(0):anchor_features.size(0) + positive_features.size(0), 0],
                    tsne_results[anchor_features.size(0):anchor_features.size(0) + positive_features.size(0), 1], c='g', label='Positive')
        plt.scatter(tsne_results[anchor_features.size(0) + positive_features.size(0):, 0],
                    tsne_results[anchor_features.size(0) + positive_features.size(0):, 1], c='b', label='Negative')
        plt.legend()
        plt.title(f'TSNE {stage} Epoch {epoch}')
        tsne_dir = os.path.join(self.save_dir, f'epoch{epoch:03d}')
        os.makedirs(tsne_dir, exist_ok=True)
        tsne_path = os.path.join(tsne_dir, f'{stage}_tsne_epoch{epoch:03d}.png')
        plt.savefig(tsne_path)
        plt.close()

        self.logger.experiment.log({f"{stage}_tsne": wandb.Image(tsne_path)})

    def save_model(self, epoch_label):
        model_path = os.path.join(self.save_dir, f'epoch{epoch_label:03d}.pt')
        torch.save(self.state_dict(), model_path)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['epoch'] = self.current_epoch
        checkpoint['global_step'] = self.global_step

    def on_load_checkpoint(self, checkpoint):
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
