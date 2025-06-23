import os
import torch
import pytorch_lightning as pl
from torchmetrics import AUROC

class AETrainer(pl.LightningModule):
    def __init__(self, feature_extractor, autoencoder, lr=1e-3, wd=1e-5, save_dir="checkpoints"):
        super(AETrainer, self).__init__()
        self.feature_extractor = feature_extractor
        self.autoencoder = autoencoder
        self.lr = lr
        self.wd = wd
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.save_dir = save_dir
        self.train_outputs = []
        self.val_outputs = []
        self.train_auroc = AUROC(task='binary')
        self.val_auroc = AUROC(task='binary')
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        autoencoder_output = self.autoencoder(features)
        if isinstance(autoencoder_output, dict):
            return autoencoder_output['output'], features
        return autoencoder_output, features

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        with torch.no_grad():
            features = self.feature_extractor(inputs).detach()
        reconstructed, _ = self.forward(inputs)
        loss = self.criterion(features, reconstructed).mean()
        self.log('train_loss', loss, prog_bar=True)
        self.train_outputs.append((features.cpu(), reconstructed.cpu(), labels.cpu()))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        with torch.no_grad():
            features = self.feature_extractor(inputs).detach()
        reconstructed, _ = self.forward(inputs)
        loss = self.criterion(features, reconstructed).mean()
        self.log('val_loss', loss, prog_bar=True)
        self.val_outputs.append((features.cpu(), reconstructed.cpu(), labels.cpu()))
        return loss

    def on_train_epoch_end(self):
        self.calculate_and_log_metrics(self.train_outputs, "train")
        self.train_outputs = []
        self.save_model(self.current_epoch)

    def on_validation_epoch_end(self):
        self.calculate_and_log_metrics(self.val_outputs, "val")
        self.val_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    def calculate_and_log_metrics(self, outputs, stage):
        features_list = []
        reconstructed_list = []
        labels_list = []

        for features, reconstructed, labels in outputs:
            features_list.append(features)
            reconstructed_list.append(reconstructed)
            labels_list.append(labels)

        features = torch.cat(features_list)
        reconstructed = torch.cat(reconstructed_list)
        labels = torch.cat(labels_list)

        errors = self.criterion(reconstructed, features).mean(dim=1)
        if stage == "train":
            auroc = self.train_auroc(errors, labels)
        else:
            auroc = self.val_auroc(errors, labels)

        self.log(f'{stage}_auc', auroc, prog_bar=True)

    def save_model(self, epoch_label):
        model_path = os.path.join(self.save_dir, f'epoch{epoch_label:03d}.ckpt')
        self.trainer.save_checkpoint(model_path)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['epoch'] = self.current_epoch
        checkpoint['global_step'] = self.global_step

    def on_load_checkpoint(self, checkpoint):
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

