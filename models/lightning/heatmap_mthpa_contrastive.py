import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
import numpy as np
import gc


class HeatmapMTHPAContrastive(pl.LightningModule):
    def __init__(self, 
                 mthpa_model, 
                 lr=1e-4, 
                 wd=1e-5, 
                 save_dir="checkpoints", 
                 temperature=0.07,
                 gradient_accumulation_steps=1,
                 use_mixed_precision=False,
                 visualize_every_n_epochs=1):  # TSNE 시각화 주기
        super().__init__()
        self.mthpa_model = mthpa_model
        self.lr = lr
        self.wd = wd
        self.save_dir = save_dir
        self.temperature = temperature
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        self.visualize_every_n_epochs = visualize_every_n_epochs
        
        # Feature storage for visualization
        self.train_features = None
        self.val_features = None
        
        # Manual optimization for gradient accumulation if needed
        if gradient_accumulation_steps > 1:
            self.automatic_optimization = False
        
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, x, mask=None):
        # x: (batch_size, time, objects, 224, 224)
        return self.mthpa_model(x, mask)

    def info_nce_loss(self, anchor, pair, labels, temperature):
        # Flatten features to (B, D)
        anchor = anchor.view(anchor.size(0), -1)
        pair = pair.view(pair.size(0), -1)
        
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)
            pair = pair.unsqueeze(0)
            labels = labels.unsqueeze(0)
            
        # Normalize features
        anchor = F.normalize(anchor, dim=-1)
        pair = F.normalize(pair, dim=-1)
        
        # Compute similarity
        sim = torch.sum(anchor * pair, dim=-1) / temperature  # (B,)
        
        # Binary cross entropy loss for contrastive learning
        loss_pos = -labels * torch.log(torch.sigmoid(sim) + 1e-12)
        loss_neg = -(1 - labels) * torch.log(1 - torch.sigmoid(sim) + 1e-12)
        loss = (loss_pos + loss_neg).mean()
        
        return loss

    def compute_loss(self, anchor_feat, pair_feat, labels):
        return self.info_nce_loss(anchor_feat, pair_feat, labels, self.temperature)

    def training_step(self, batch, batch_idx):
        # Unpack batch with masks
        if len(batch) == 5:
            anchor, pair, labels, anchor_mask, pair_mask = batch
        else:
            # Backward compatibility
            anchor, pair, labels = batch
            anchor_mask = pair_mask = None
        
        if self.gradient_accumulation_steps > 1:
            # Manual optimization
            opt = self.optimizers()
            
            # Clear gradients at the start of accumulation
            if batch_idx % self.gradient_accumulation_steps == 0:
                opt.zero_grad(set_to_none=True)
            
            # Forward pass with masks
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    anchor_feat = self.mthpa_model(anchor, anchor_mask)
                    pair_feat = self.mthpa_model(pair, pair_mask)
                    loss = self.compute_loss(anchor_feat, pair_feat, labels)
                    loss = loss / self.gradient_accumulation_steps
            else:
                anchor_feat = self.mthpa_model(anchor, anchor_mask)
                pair_feat = self.mthpa_model(pair, pair_mask)
                loss = self.compute_loss(anchor_feat, pair_feat, labels)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            self.manual_backward(loss)
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                opt.step()
            
            # Scale loss back for logging
            loss = loss * self.gradient_accumulation_steps
        else:
            # Automatic optimization with masks
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    anchor_feat = self.mthpa_model(anchor, anchor_mask)
                    pair_feat = self.mthpa_model(pair, pair_mask)
                    loss = self.compute_loss(anchor_feat, pair_feat, labels)
            else:
                anchor_feat = self.mthpa_model(anchor, anchor_mask)
                pair_feat = self.mthpa_model(pair, pair_mask)
                loss = self.compute_loss(anchor_feat, pair_feat, labels)
        
        # Compute accuracy
        with torch.no_grad():
            anchor_norm = F.normalize(anchor_feat.detach().view(anchor_feat.size(0), -1), dim=-1)
            pair_norm = F.normalize(pair_feat.detach().view(pair_feat.size(0), -1), dim=-1)
            sim = torch.sum(anchor_norm * pair_norm, dim=-1)
            pred = (sim > 0).float()
            acc = (pred == labels).float().mean()
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, 
                 logger=True, batch_size=anchor.size(0), sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=anchor.size(0), sync_dist=True)
        
        # 특정 epoch에만 feature 저장 (TSNE 시각화용)
        if self.current_epoch % self.visualize_every_n_epochs == 0:
            with torch.no_grad():
                # Store only anchor features to save memory
                anchor_flat = anchor_feat.detach().cpu().view(anchor_feat.size(0), -1)
                batch_labels = labels.detach().cpu()
                
                # Skip if batch is empty or features are invalid
                if anchor_flat.size(0) == 0 or anchor_flat.size(1) == 0:
                    return loss
                
                if self.train_features is None:
                    self.train_features = (anchor_flat, batch_labels)
                else:
                    # Append all features (no sampling)
                    existing_features, existing_labels = self.train_features
                    self.train_features = (
                        torch.cat([existing_features, anchor_flat], dim=0),
                        torch.cat([existing_labels, batch_labels], dim=0)
                    )
                    
                    # 디버깅용 출력 (첫 epoch에서만)
                    if self.current_epoch == 0 and batch_idx % 100 == 0:
                        print(f"[Feature Storage] Epoch {self.current_epoch}, Batch {batch_idx}: "
                              f"Total stored {self.train_features[0].size(0)} features")
        
        # Memory cleanup
        del anchor_feat, pair_feat
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch with masks
        if len(batch) == 5:
            anchor, pair, labels, anchor_mask, pair_mask = batch
        else:
            # Backward compatibility
            anchor, pair, labels = batch
            anchor_mask = pair_mask = None
        
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    anchor_feat = self.mthpa_model(anchor, anchor_mask)
                    pair_feat = self.mthpa_model(pair, pair_mask)
                    loss = self.compute_loss(anchor_feat, pair_feat, labels)
            else:
                anchor_feat = self.mthpa_model(anchor, anchor_mask)
                pair_feat = self.mthpa_model(pair, pair_mask)
                loss = self.compute_loss(anchor_feat, pair_feat, labels)
            
            # Compute accuracy
            anchor_norm = F.normalize(anchor_feat.view(anchor_feat.size(0), -1), dim=-1)
            pair_norm = F.normalize(pair_feat.view(pair_feat.size(0), -1), dim=-1)
            sim = torch.sum(anchor_norm * pair_norm, dim=-1)
            pred = (sim > 0).float()
            acc = (pred == labels).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, 
                 logger=True, batch_size=anchor.size(0), sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=anchor.size(0), sync_dist=True)
        
        # 특정 epoch에만 validation feature 저장
        if self.current_epoch % self.visualize_every_n_epochs == 0:
            with torch.no_grad():
                # Store only anchor features
                anchor_flat = anchor_feat.cpu().view(anchor_feat.size(0), -1)
                
                if self.val_features is None:
                    self.val_features = (anchor_flat, labels.cpu())
                else:
                    # Append all features
                    all_feats, all_labs = self.val_features
                    self.val_features = (
                        torch.cat([all_feats, anchor_flat], dim=0),
                        torch.cat([all_labs, labels.cpu()], dim=0)
                    )
        
        # Memory cleanup
        del anchor_feat, pair_feat
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
        
        return loss

    def on_train_epoch_end(self):
        # Visualize features only on specific epochs
        if self.current_epoch % self.visualize_every_n_epochs == 0 and self.train_features is not None:
            try:
                features, labels = self.train_features
                print(f"[TSNE] Train features shape: {features.shape} (full training set)")
                if features.shape[0] >= 10:
                    self.visualize_tsne(features, labels, "train", self.current_epoch)
                else:
                    print(f"[TSNE] Skipping visualization - too few samples ({features.shape[0]})")
            except Exception as e:
                print(f"Failed to visualize train features: {e}")
            finally:
                self.train_features = None
        elif self.train_features is not None:
            # Clear features if not visualizing this epoch
            self.train_features = None
        
        # Save model periodically
        if self.current_epoch % 5 == 0:
            self.save_model(self.current_epoch)
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        # Visualize features only on specific epochs
        if self.current_epoch % self.visualize_every_n_epochs == 0 and self.val_features is not None:
            try:
                features, labels = self.val_features
                print(f"[TSNE] Val features shape: {features.shape} (full validation set)")
                if features.shape[0] >= 10:
                    self.visualize_tsne(features, labels, "val", self.current_epoch)
                else:
                    print(f"[TSNE] Skipping visualization - too few samples ({features.shape[0]})")
            except Exception as e:
                print(f"Failed to visualize val features: {e}")
            finally:
                self.val_features = None
        elif self.val_features is not None:
            # Clear features if not visualizing this epoch
            self.val_features = None
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    def on_train_start(self):
        # Initialize with a high value for val_loss
        self.log('val_loss', float('inf'), prog_bar=True, logger=True, sync_dist=True)

    def visualize_tsne(self, features, labels, stage, epoch):
        """Memory-efficient TSNE visualization"""
        try:
            features = features.numpy()
            labels = labels.numpy()
            features = features.reshape(features.shape[0], -1)
            
            if features.shape[1] < 2:
                print(f"[TSNE] Skipping TSNE visualization for {stage} at epoch {epoch} because feature dim < 2")
                return
                
            num_samples = features.shape[0]
            # perplexity를 샘플 수에 맞게 조정 (최대 50)
            tsne_perplexity = min(50, max(5, num_samples // 4))
            
            if num_samples <= tsne_perplexity:
                tsne_perplexity = max(2, num_samples - 1)
                print(f"[TSNE] Adjusted perplexity to {tsne_perplexity} due to small sample size")
            
            print(f"[TSNE] Creating visualization for {stage} at epoch {epoch} with {num_samples} samples (perplexity={tsne_perplexity})")
            
            # 더 안정적인 TSNE 설정
            tsne = TSNE(
                n_components=2, 
                perplexity=tsne_perplexity, 
                random_state=42,
                n_iter=1000,  # 더 많은 iteration으로 수렴 개선
                init='pca',   # PCA 초기화로 안정성 향상
                learning_rate='auto'
            )
            tsne_results = tsne.fit_transform(features)

            # Plot with color by label
            plt.figure(figsize=(10, 6))
            
            # 클래스별 샘플 수 계산
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_info = dict(zip(unique_labels, counts))
            
            for label, color, name in zip([0, 1], ['green', 'red'], ['normal', 'abnormal']):
                idx = labels == label
                if np.any(idx):
                    count = label_info.get(label, 0)
                    plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], 
                              c=color, label=f'{name} (n={count})', 
                              alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
            
            plt.legend()
            plt.title(f't-SNE {stage} Epoch {epoch} (Total: {num_samples} samples)')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True, alpha=0.3)
            
            # Save path
            tsne_dir = os.path.join(self.save_dir, f'tsne_visualization')
            os.makedirs(tsne_dir, exist_ok=True)
            tsne_path = os.path.join(tsne_dir, f'tsne_{stage}_epoch_{epoch}.png')
            plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Log to wandb if available
            if hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
                try:
                    self.logger.experiment.log({f"{stage}_tsne": wandb.Image(tsne_path)})
                except Exception as e:
                    print(f"[TSNE] Failed to log to wandb: {e}")
                
            print(f"[TSNE] Saved visualization to: {tsne_path}")
                    
        except Exception as e:
            print(f"[TSNE] Visualization failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close('all')
            gc.collect()

    def save_model(self, epoch_label):
        """Save model checkpoint"""
        try:
            model_path = os.path.join(self.save_dir, f'epoch{epoch_label:03d}.ckpt')
            self.trainer.save_checkpoint(model_path)
            print(f"Saved checkpoint: {model_path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def on_save_checkpoint(self, checkpoint):
        # Save model state dict
        checkpoint['model_state_dict'] = self.mthpa_model.state_dict()
        # Save additional info
        checkpoint['epoch'] = self.current_epoch
        checkpoint['global_step'] = self.global_step

    def on_load_checkpoint(self, checkpoint):
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            self.mthpa_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.wd,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs if hasattr(self, 'trainer') and self.trainer else 100, 
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            },
        }