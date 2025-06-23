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
        anchor, positive, negative, labels = batch  # is_normal -> labels
        anchor_features = self.heatmap_c3d(anchor)
        positive_features = self.heatmap_c3d(positive)
        negative_features = self.heatmap_c3d(negative)

        loss = self.triplet_loss(anchor_features, positive_features, negative_features)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=anchor.size(0),
                 sync_dist=True)

        if self.train_features is None:
            self.train_features = (
                anchor_features.detach().cpu(), positive_features.detach().cpu(), negative_features.detach().cpu(),
                labels.detach().cpu()  # is_normal -> labels
            )
        else:
            self.train_features = (
                torch.cat((self.train_features[0], anchor_features.detach().cpu()), dim=0),
                torch.cat((self.train_features[1], positive_features.detach().cpu()), dim=0),
                torch.cat((self.train_features[2], negative_features.detach().cpu()), dim=0),
                torch.cat((self.train_features[3], labels.detach().cpu()), dim=0)  # is_normal -> labels
            )

        return loss

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative, labels = batch  # is_normal -> labels
        anchor_features = self.heatmap_c3d(anchor)
        positive_features = self.heatmap_c3d(positive)
        negative_features = self.heatmap_c3d(negative)

        loss = self.triplet_loss(anchor_features, positive_features, negative_features)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=anchor.size(0),
                 sync_dist=True)

        if self.val_features is None:
            self.val_features = (
                anchor_features.detach().cpu(), positive_features.detach().cpu(), negative_features.detach().cpu(),
                labels.detach().cpu()  # is_normal -> labels
            )
        else:
            self.val_features = (
                torch.cat((self.val_features[0], anchor_features.detach().cpu()), dim=0),
                torch.cat((self.val_features[1], positive_features.detach().cpu()), dim=0),
                torch.cat((self.val_features[2], negative_features.detach().cpu()), dim=0),
                torch.cat((self.val_features[3], labels.detach().cpu()), dim=0)  # is_normal -> labels
            )

        return loss

    def on_train_epoch_end(self, unused=None):
        # Validation epoch일 때만 TSNE 및 Temporal Saliency 시각화 수행
        if (self.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0:
            if self.train_features is not None:
                labels = self.train_features[3]
                class_names = ["assault", "falldown", "normal"]

                for class_idx, class_name in enumerate(class_names):
                    mask = labels == class_idx  # 해당 클래스의 마스크 생성
                    class_features = (
                        self.train_features[0][mask],  # anchor
                        self.train_features[1][mask],  # positive
                        self.train_features[2][mask]  # negative
                    )

                    # TSNE 시각화
                    self.visualize_tsne(*class_features, f"train_{class_name}", self.current_epoch)

                # Grad-CAM 활성화 맵 계산
                cams = self.heatmap_c3d.generate_cam()  # (3, frame_count, 224, 224)
                temporal_saliency = self.compute_temporal_saliency(cams)  # (3, frame_count)

                # 각 클래스별 Temporal Saliency 시각화
                for class_idx, class_name in enumerate(class_names):
                    self.visualize_temporal_saliency(temporal_saliency[class_idx], class_name, self.current_epoch)

                self.train_features = None  # epoch 끝나면 초기화

        self.save_model(self.current_epoch)

    def on_validation_epoch_end(self):
        if self.val_features is not None:
            labels = self.val_features[3]
            class_names = ["assault", "falldown", "normal"]

            for class_idx, class_name in enumerate(class_names):
                mask = labels == class_idx  # 해당 클래스의 마스크 생성
                class_features = (
                    self.val_features[0][mask],  # anchor
                    self.val_features[1][mask],  # positive
                    self.val_features[2][mask]  # negative
                )

                # TSNE 시각화
                self.visualize_tsne(*class_features, f"val_{class_name}", self.current_epoch)

            # Grad-CAM 활성화 맵 계산
            cams = self.heatmap_c3d.generate_cam()  # (3, frame_count, 224, 224)
            temporal_saliency = self.compute_temporal_saliency(cams)  # (3, frame_count)

            # 각 클래스별 Temporal Saliency 시각화
            for class_idx, class_name in enumerate(class_names):
                self.visualize_temporal_saliency(temporal_saliency[class_idx], class_name, self.current_epoch)

            self.val_features = None  # epoch 끝나면 초기화

    def on_train_start(self):
        self.log('val_loss', float('inf'), prog_bar=True, logger=True, sync_dist=True)

    def compute_temporal_saliency(self, cams):
        temporal_saliency = torch.mean(cams, dim=(2, 3))  # 공간 축(224, 224)에 대해 평균

        return temporal_saliency  # (class_count, frame_count)

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
        # plt.scatter(tsne_results[:anchor_features.size(0), 0], tsne_results[:anchor_features.size(0), 1], c='r', label='Anchor')
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

    def visualize_temporal_saliency(self, saliency, class_name, epoch):
        """
        Temporal Saliency를 시각화하여 저장하는 함수
        Args:
        - saliency: (frame_count,) 형태의 각 프레임의 중요도를 나타내는 배열
        - class_name: 클래스 이름 (예: "assault", "falldown", "normal")
        - epoch: 현재 에포크
        """
        plt.figure(figsize=(10, 4))
        plt.plot(saliency.cpu().numpy())
        plt.title(f'Temporal Saliency for {class_name} at Epoch {epoch}')
        plt.xlabel('Frame')
        plt.ylabel('Saliency')
        plt.grid(True)

        # 저장 경로 설정
        ts_dir = os.path.join(self.save_dir, f'epoch{epoch:03d}')
        os.makedirs(ts_dir, exist_ok=True)
        ts_path = os.path.join(ts_dir, f'{class_name}_temporal_saliency_epoch{epoch:03d}.png')

        # 시각화된 그래프 저장
        plt.savefig(ts_path)
        plt.close()

        # 로그에 추가 (옵션)
        self.logger.experiment.log({f"{class_name}_temporal_saliency": wandb.Image(ts_path)})

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

    def save_model(self, epoch_label):
        model_path = os.path.join(self.save_dir, f'epoch{epoch_label:03d}.ckpt')
        self.trainer.save_checkpoint(model_path)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['epoch'] = self.hparams.current_epoch
        checkpoint['global_step'] = self.hparams.global_step

    def on_load_checkpoint(self, checkpoint):
        self.hparams.current_epoch = checkpoint['epoch']
        self.hparams.global_step = checkpoint['global_step']

