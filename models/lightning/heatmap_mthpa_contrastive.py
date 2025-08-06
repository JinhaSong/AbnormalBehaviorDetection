import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
import numpy as np
import gc
from tqdm import tqdm


class HeatmapMTHPAContrastive(pl.LightningModule):
    def __init__(self, 
                 mthpa_model, 
                 lr=1e-4, 
                 wd=1e-5, 
                 save_dir="checkpoints", 
                 temperature=0.07,
                 gradient_accumulation_steps=1,
                 use_mixed_precision=False,
                 visualize_every_n_epochs=5,
                 max_tsne_samples=1000,  # 새로운 파라미터: TSNE에 사용할 최대 샘플 수
                 tsne_batch_size=32):    # 새로운 파라미터: TSNE inference 배치 크기
        super().__init__()
        self.mthpa_model = mthpa_model
        self.lr = lr
        self.wd = wd
        self.save_dir = save_dir
        self.temperature = temperature
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        self.visualize_every_n_epochs = visualize_every_n_epochs
        self.max_tsne_samples = max_tsne_samples
        self.tsne_batch_size = tsne_batch_size
        
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
        # Unpack batch
        if len(batch) == 6:
            anchor, pair, labels, anchor_indices, anchor_mask, pair_mask = batch
        elif len(batch) == 5:
            anchor, pair, labels, anchor_mask, pair_mask = batch
            anchor_indices = None
        else:
            anchor, pair, labels = batch
            anchor_mask = pair_mask = None
            anchor_indices = None
        
        if self.gradient_accumulation_steps > 1:
            opt = self.optimizers()
            if batch_idx % self.gradient_accumulation_steps == 0:
                opt.zero_grad(set_to_none=True)
            
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
            
            self.manual_backward(loss)
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                opt.step()
            
            loss = loss * self.gradient_accumulation_steps
        else:
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
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, 
                 logger=True, batch_size=anchor.size(0), sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=anchor.size(0), sync_dist=True)
        
        # Memory cleanup
        del anchor_feat, pair_feat
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        if len(batch) == 6:
            anchor, pair, labels, anchor_indices, anchor_mask, pair_mask = batch
        elif len(batch) == 5:
            anchor, pair, labels, anchor_mask, pair_mask = batch
            anchor_indices = None
        else:
            anchor, pair, labels = batch
            anchor_mask = pair_mask = None
            anchor_indices = None
        
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
        
        # Memory cleanup
        del anchor_feat, pair_feat
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
        
        return loss

    def collect_tsne_features(self, dataloader, stage, max_samples=None):
        """메모리 효율적인 TSNE feature 수집"""
        if max_samples is None:
            max_samples = self.max_tsne_samples
            
        # Set model to eval mode
        self.eval()
        
        all_features = []
        all_labels = []
        seen_indices = set()
        total_collected = 0
        
        # First, collect unique samples up to max_samples
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"[TSNE] Collecting {stage} features")):
                if total_collected >= max_samples:
                    break
                    
                # Unpack batch
                if len(batch) == 6:
                    anchor, pair, labels, anchor_indices, anchor_mask, pair_mask = batch
                elif len(batch) == 5:
                    anchor, pair, labels, anchor_mask, pair_mask = batch
                    anchor_indices = torch.arange(batch_idx * anchor.size(0), 
                                                (batch_idx + 1) * anchor.size(0))
                else:
                    anchor, pair, labels = batch
                    anchor_mask = pair_mask = None
                    anchor_indices = torch.arange(batch_idx * anchor.size(0), 
                                                (batch_idx + 1) * anchor.size(0))
                
                # Process in smaller sub-batches to save memory
                sub_batch_size = min(self.tsne_batch_size, anchor.size(0))
                
                for start_idx in range(0, anchor.size(0), sub_batch_size):
                    if total_collected >= max_samples:
                        break
                        
                    end_idx = min(start_idx + sub_batch_size, anchor.size(0))
                    
                    # Get sub-batch
                    anchor_sub = anchor[start_idx:end_idx].to(self.device)
                    labels_sub = labels[start_idx:end_idx]
                    indices_sub = anchor_indices[start_idx:end_idx]
                    
                    if anchor_mask is not None:
                        anchor_mask_sub = anchor_mask[start_idx:end_idx].to(self.device)
                    else:
                        anchor_mask_sub = None
                    
                    # Check for unique samples
                    unique_mask = []
                    for i, idx in enumerate(indices_sub):
                        idx_value = idx.item() if torch.is_tensor(idx) else idx
                        if idx_value not in seen_indices:
                            seen_indices.add(idx_value)
                            unique_mask.append(i)
                    
                    if not unique_mask:
                        del anchor_sub
                        if anchor_mask_sub is not None:
                            del anchor_mask_sub
                        torch.cuda.empty_cache()
                        continue
                    
                    # Process only unique samples
                    unique_mask = torch.tensor(unique_mask)
                    anchor_unique = anchor_sub[unique_mask]
                    labels_unique = labels_sub[unique_mask]
                    
                    if anchor_mask_sub is not None:
                        anchor_mask_unique = anchor_mask_sub[unique_mask]
                    else:
                        anchor_mask_unique = None
                    
                    # Get features
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            features = self.mthpa_model(anchor_unique, anchor_mask_unique)
                    else:
                        features = self.mthpa_model(anchor_unique, anchor_mask_unique)
                    
                    # Move to CPU immediately and flatten
                    features = features.detach().cpu()
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)
                    
                    all_features.append(features)
                    all_labels.append(labels_unique.cpu())
                    
                    total_collected += features.size(0)
                    
                    # Aggressive memory cleanup
                    del anchor_sub, anchor_unique, features
                    if anchor_mask_sub is not None:
                        del anchor_mask_sub, anchor_mask_unique
                    torch.cuda.empty_cache()
                
                # More aggressive cleanup after each batch
                if batch_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # Set model back to train mode if needed
        if stage == "train":
            self.train()
        
        # Concatenate all features
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Random sampling if we still have too many samples
            if all_features.size(0) > max_samples:
                indices = torch.randperm(all_features.size(0))[:max_samples]
                all_features = all_features[indices]
                all_labels = all_labels[indices]
            
            return all_features, all_labels
        else:
            return None, None

    def on_train_epoch_end(self):
        # TSNE 시각화 (특정 epoch에만)
        if self.current_epoch % self.visualize_every_n_epochs == 0:
            if self.trainer.is_global_zero:
                try:
                    # Clear cache before starting
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Get train dataloader
                    train_dataloader = self.trainer.train_dataloader
                    
                    # Collect features with memory limit
                    features, labels = self.collect_tsne_features(train_dataloader, "train")
                    
                    if features is not None and features.size(0) >= 10:
                        # Separate normal and abnormal
                        normal_mask = labels == 0
                        abnormal_mask = labels == 1
                        
                        normal_features = features[normal_mask]
                        abnormal_features = features[abnormal_mask]
                        
                        print(f"[TSNE] Train features collected: {features.size(0)} samples "
                              f"(Normal: {normal_features.size(0)}, Abnormal: {abnormal_features.size(0)})")
                        
                        # Create visualization
                        self.visualize_tsne(normal_features, abnormal_features, "train", self.current_epoch)
                        
                        # Cleanup
                        del features, labels, normal_features, abnormal_features
                    else:
                        print(f"[TSNE] Skipping train visualization - insufficient samples")
                    
                except Exception as e:
                    print(f"Failed to create train TSNE: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # Save model periodically
        if self.current_epoch % 5 == 0:
            self.save_model(self.current_epoch)

    def on_validation_epoch_end(self):
        # TSNE 시각화 (특정 epoch에만)
        if self.current_epoch % self.visualize_every_n_epochs == 0:
            if self.trainer.is_global_zero and self.current_epoch != 0:
                try:
                    # Clear cache before starting
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Get val dataloader
                    val_dataloader = self.trainer.val_dataloaders[0] if isinstance(self.trainer.val_dataloaders, list) else self.trainer.val_dataloaders
                    
                    # Collect features with memory limit
                    features, labels = self.collect_tsne_features(val_dataloader, "val")
                    
                    if features is not None and features.size(0) >= 10:
                        # Separate normal and abnormal
                        normal_mask = labels == 0
                        abnormal_mask = labels == 1
                        
                        normal_features = features[normal_mask]
                        abnormal_features = features[abnormal_mask]
                        
                        print(f"[TSNE] Val features collected: {features.size(0)} samples "
                              f"(Normal: {normal_features.size(0)}, Abnormal: {abnormal_features.size(0)})")
                        
                        # Create visualization
                        self.visualize_tsne(normal_features, abnormal_features, "val", self.current_epoch)
                        
                        # Cleanup
                        del features, labels, normal_features, abnormal_features
                    else:
                        print(f"[TSNE] Skipping val visualization - insufficient samples")
                    
                except Exception as e:
                    print(f"Failed to create val TSNE: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

    def visualize_tsne(self, normal_features, abnormal_features, stage, epoch):
        """TSNE visualization for normal and abnormal features"""
        # Handle empty tensors
        if normal_features.numel() == 0 and abnormal_features.numel() == 0:
            print(f"Skipping TSNE visualization for {stage} at epoch {epoch} - no features")
            return
        
        # Flatten features if needed
        if normal_features.numel() > 0 and normal_features.dim() > 2:
            normal_features = normal_features.view(normal_features.size(0), -1)
        if abnormal_features.numel() > 0 and abnormal_features.dim() > 2:
            abnormal_features = abnormal_features.view(abnormal_features.size(0), -1)
            
        # Prepare features list
        feature_list = []
        if normal_features.numel() > 0:
            feature_list.append(normal_features)
        if abnormal_features.numel() > 0:
            feature_list.append(abnormal_features)
            
        # Concatenate all features
        features = torch.cat(feature_list, dim=0).numpy()
        num_samples = features.shape[0]
        tsne_perplexity = min(30, max(1, (num_samples - 1) // 3))

        if num_samples <= tsne_perplexity:
            print(f"Skipping TSNE visualization for {stage} at epoch {epoch} due to insufficient samples.")
            return

        # Additional memory optimization: reduce feature dimension if too large
        if features.shape[1] > 1000:
            print(f"[TSNE] Reducing feature dimension from {features.shape[1]} to 1000 using PCA")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(1000, features.shape[0] - 1))
            features = pca.fit_transform(features)
            gc.collect()

        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42, n_jobs=1)
        tsne_results = tsne.fit_transform(features)

        plt.figure(figsize=(10, 6))
        
        # Plot normal samples (green)
        if normal_features.numel() > 0:
            plt.scatter(tsne_results[:normal_features.size(0), 0], 
                       tsne_results[:normal_features.size(0), 1], 
                       c='g', label='Normal', alpha=0.7)
        
        # Plot abnormal samples (red)
        if abnormal_features.numel() > 0:
            start_idx = normal_features.size(0) if normal_features.numel() > 0 else 0
            plt.scatter(tsne_results[start_idx:, 0],
                       tsne_results[start_idx:, 1], 
                       c='r', label='Abnormal', alpha=0.7)
        
        plt.legend()
        plt.title(f'TSNE {stage} Epoch {epoch}')
        tsne_dir = os.path.join(self.save_dir, f'epoch{epoch:03d}')
        os.makedirs(tsne_dir, exist_ok=True)
        tsne_path = os.path.join(tsne_dir, f'{stage}_tsne_epoch{epoch:03d}.png')
        plt.savefig(tsne_path)
        plt.close()

        if self.logger:
            self.logger.experiment.log({f"{stage}_tsne": wandb.Image(tsne_path)})

    def on_train_start(self):
        # Initialize with a high value for val_loss
        self.log('val_loss', float('inf'), prog_bar=True, logger=True, sync_dist=True)

    def save_model(self, epoch_label):
        model_path = os.path.join(self.save_dir, f'epoch{epoch_label:03d}.ckpt')
        self.trainer.save_checkpoint(model_path)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['epoch'] = self.current_epoch
        checkpoint['global_step'] = self.global_step

    def on_load_checkpoint(self, checkpoint):
        pass

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