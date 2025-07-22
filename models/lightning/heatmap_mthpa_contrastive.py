import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
import numpy as np
import gc


class HeatmapMTHPA_Contrastive(pl.LightningModule):
    def __init__(self, 
                 mthpa_model, 
                 lr=1e-4, 
                 wd=1e-5, 
                 save_dir="checkpoints", 
                 temperature=0.07,
                 gradient_accumulation_steps=1,
                 use_mixed_precision=False,
                 max_stored_features=1000):
        super().__init__()
        self.mthpa_model = mthpa_model
        self.lr = lr
        self.wd = wd
        self.save_dir = save_dir
        self.temperature = temperature
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        self.max_stored_features = max_stored_features
        
        # Feature storage for visualization
        self.train_features = None
        self.val_features = None
        
        # Manual optimization for gradient accumulation if needed
        if gradient_accumulation_steps > 1:
            self.automatic_optimization = False
        
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

    def training_step(self, batch, batch_idx):
        anchor, pair, labels = batch  # (B, T, O, 224, 224), (B, T, O, 224, 224), (B,)
        
        if self.gradient_accumulation_steps > 1:
            # Manual optimization
            opt = self.optimizers()
            
            # Clear gradients at the start of accumulation
            if batch_idx % self.gradient_accumulation_steps == 0:
                opt.zero_grad(set_to_none=True)
            
            # Forward pass
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    anchor_feat = self.mthpa_model(anchor)
                    pair_feat = self.mthpa_model(pair)
                    loss = self.info_nce_loss(anchor_feat, pair_feat, labels, self.temperature)
                    loss = loss / self.gradient_accumulation_steps
            else:
                anchor_feat = self.mthpa_model(anchor)
                pair_feat = self.mthpa_model(pair)
                loss = self.info_nce_loss(anchor_feat, pair_feat, labels, self.temperature)
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
            # Automatic optimization
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    anchor_feat = self.mthpa_model(anchor)
                    pair_feat = self.mthpa_model(pair)
                    loss = self.info_nce_loss(anchor_feat, pair_feat, labels, self.temperature)
            else:
                anchor_feat = self.mthpa_model(anchor)
                pair_feat = self.mthpa_model(pair)
                loss = self.info_nce_loss(anchor_feat, pair_feat, labels, self.temperature)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, 
                 logger=True, batch_size=anchor.size(0), sync_dist=True)
        
        # Store features for visualization (limited)
        if self.global_step % 500 == 0:  # Store less frequently
            with torch.no_grad():
                feats = torch.cat([anchor_feat.detach().cpu(), pair_feat.detach().cpu()], dim=0)
                labs = torch.cat([labels.cpu(), labels.cpu()], dim=0)
                
                # Limit stored features
                if len(feats) > self.max_stored_features:
                    indices = torch.randperm(len(feats))[:self.max_stored_features]
                    feats = feats[indices]
                    labs = labs[indices]
                
                if self.train_features is None:
                    self.train_features = (feats, labs)
                else:
                    # Concatenate and limit total size
                    all_feats = torch.cat([self.train_features[0], feats], dim=0)
                    all_labs = torch.cat([self.train_features[1], labs], dim=0)
                    if len(all_feats) > self.max_stored_features:
                        indices = torch.randperm(len(all_feats))[:self.max_stored_features]
                        all_feats = all_feats[indices]
                        all_labs = all_labs[indices]
                    self.train_features = (all_feats, all_labs)
        
        # Memory cleanup
        del anchor_feat, pair_feat
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            
        return loss

    def validation_step(self, batch, batch_idx):
        anchor, pair, labels = batch
        
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    anchor_feat = self.mthpa_model(anchor)
                    pair_feat = self.mthpa_model(pair)
                    loss = self.info_nce_loss(anchor_feat, pair_feat, labels, self.temperature)
            else:
                anchor_feat = self.mthpa_model(anchor)
                pair_feat = self.mthpa_model(pair)
                loss = self.info_nce_loss(anchor_feat, pair_feat, labels, self.temperature)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, 
                 logger=True, batch_size=anchor.size(0), sync_dist=True)
        
        # Store features for visualization (limited to first few batches)
        if batch_idx < 10:  # Only first 10 batches
            feats = torch.cat([anchor_feat.cpu(), pair_feat.cpu()], dim=0)
            labs = torch.cat([labels.cpu(), labels.cpu()], dim=0)
            
            if self.val_features is None:
                self.val_features = (feats, labs)
            else:
                all_feats = torch.cat([self.val_features[0], feats], dim=0)
                all_labs = torch.cat([self.val_features[1], labs], dim=0)
                # Limit total stored features
                if len(all_feats) > self.max_stored_features:
                    all_feats = all_feats[:self.max_stored_features]
                    all_labs = all_labs[:self.max_stored_features]
                self.val_features = (all_feats, all_labs)
        
        # Memory cleanup
        del anchor_feat, pair_feat
        torch.cuda.empty_cache()
        
        return loss

    def on_train_epoch_end(self):
        # Visualize features
        if self.train_features is not None:
            try:
                features, labels = self.train_features
                self.visualize_tsne(features, labels, "train", self.current_epoch)
            except Exception as e:
                print(f"Failed to visualize train features: {e}")
            finally:
                self.train_features = None
        
        # Save model periodically
        if self.current_epoch % 5 == 0:
            self.save_model(self.current_epoch)
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        # Visualize features
        if self.val_features is not None:
            try:
                features, labels = self.val_features
                self.visualize_tsne(features, labels, "val", self.current_epoch)
            except Exception as e:
                print(f"Failed to visualize val features: {e}")
            finally:
                self.val_features = None
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    def on_train_start(self):
        self.log('val_loss', float('inf'), prog_bar=True, logger=True, sync_dist=True)

    def visualize_tsne(self, features, labels, stage, epoch):
        """Memory-efficient TSNE visualization"""
        try:
            features = features.numpy()
            labels = labels.numpy()
            
            # Ensure features are 2D
            if features.ndim > 2:
                features = features.reshape(features.shape[0], -1)
            
            # Check if we have enough samples
            if features.shape[0] < 10:
                print(f"[TSNE] Skipping visualization for {stage} at epoch {epoch} - too few samples")
                return
            
            num_samples = features.shape[0]
            
            # Subsample if too many features for TSNE
            if num_samples > 500:
                indices = np.random.choice(num_samples, 500, replace=False)
                features = features[indices]
                labels = labels[indices]
                num_samples = 500
            
            # Calculate perplexity
            tsne_perplexity = min(30, max(5, (num_samples - 1) // 3))
            
            if num_samples <= tsne_perplexity:
                print(f"Skipping TSNE visualization for {stage} at epoch {epoch} due to insufficient samples.")
                return
            
            # Run TSNE
            tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42, n_jobs=1)
            tsne_results = tsne.fit_transform(features)

            # Plot with color by label
            plt.figure(figsize=(8, 6))
            for label, color, name in zip([0, 1], ['blue', 'red'], ['normal', 'abnormal']):
                idx = labels == label
                if idx.any():
                    plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], 
                              c=color, label=name, alpha=0.6, s=20)
            
            plt.legend()
            plt.title(f't-SNE {stage} Epoch {epoch}')
            
            # Save figure
            tsne_dir = os.path.join(self.save_dir, 'tsne_visualization')
            os.makedirs(tsne_dir, exist_ok=True)
            tsne_path = os.path.join(tsne_dir, f'tsne_{stage}_epoch_{epoch}.png')
            plt.savefig(tsne_path, dpi=100)
            plt.close()

            # Log to wandb if available
            if hasattr(self, 'logger') and self.logger is not None:
                if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
                    self.logger.experiment.log({
                        f"tsne_{stage}": wandb.Image(tsne_path), 
                        "epoch": epoch
                    })
                    
        except Exception as e:
            print(f"TSNE visualization failed: {e}")
        finally:
            plt.close('all')
            gc.collect()

    def save_model(self, epoch_label):
        """Save model checkpoint"""
        try:
            model_path = os.path.join(self.save_dir, f'epoch{epoch_label:03d}.ckpt')
            self.trainer.save_checkpoint(model_path)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_state_dict'] = self.mthpa_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if 'model_state_dict' in checkpoint:
            self.mthpa_model.load_state_dict(checkpoint['model_state_dict'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.wd,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
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