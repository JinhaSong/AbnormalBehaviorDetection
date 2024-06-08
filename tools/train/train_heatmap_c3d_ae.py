import os
import sys
import wandb
import signal
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.file import get_unique_proj_dir_name
from models.dataset.h5_heatmap_dataset import get_transform, HDF5TrainDataset, HDF5TestDataset
from models.c3d.heatmap_c3d import create_heatmap_c3d_model
from models.ae.autoencoder import AutoEncoder
from models.vad.heatmap_c3d_ae_vad import HeatmapC3D_AE_VAD

def handle_sigint(signal, frame):
    print("Training interrupted. Closing WandB session.")
    wandb.finish()
    exit(0)

signal.signal(signal.SIGINT, handle_sigint)

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate VAD model")
    parser.add_argument('--T', type=int, default=16, help="Number of frames")
    parser.add_argument('--num-workers', type=int, default=2, help="Number of data loading workers")
    parser.add_argument('--gpus', type=str, default="0", help="Comma-separated list of GPU ids to use")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size")
    parser.add_argument('--max-epochs', type=int, default=300, help="Maximum number of epochs")
    parser.add_argument('--val-epochs', type=int, default=10, help="Validation epochs")
    parser.add_argument('--lr', type=float, default=0.00001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.00005, help="Weight decay")
    parser.add_argument('--dataset', choices=["cuhk", "shanghaitech", "ubnormal"], required=True, help="Type of the dataset")
    parser.add_argument('--dataset-dir', type=str, required=True, help="Directory path to the training HDF5 file")
    parser.add_argument('--model-dir', type=str, required=True, help="Directory path to save model checkpoints")
    parser.add_argument('--resume_from_checkpoint', type=str, help="Path to a checkpoint to resume from", default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    project_name = f"VAD_{args.dataset}"
    project_dir = get_unique_proj_dir_name(args.model_dir, f"{args.dataset}_heatmap_f{args.T}")
    run_name = f"{project_name}_heatmap_f{args.T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    test_h5_file_path = os.path.join(args.dataset_dir, f"{args.dataset}_test_heatmap_f{args.T}.h5")
    train_h5_file_path = os.path.join(args.dataset_dir, f"{args.dataset}_train_heatmap_f{args.T}.h5")

    test_dataset = HDF5TestDataset(test_h5_file_path, transform=get_transform())
    train_dataset = HDF5TrainDataset(train_h5_file_path, transform=get_transform())
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    heatmap_c3d = create_heatmap_c3d_model(args.T)
    autoencoder = AutoEncoder()
    vad_model = HeatmapC3D_AE_VAD(heatmap_c3d, autoencoder, args.lr, args.wd)

    tb_logger = TensorBoardLogger("tb_logs", name=project_name)
    wandb_logger = WandbLogger(project=project_name, name=run_name)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=project_dir,
        filename='{epoch:02d}-{val_loss:.2f}.pt',
        save_top_k=1,
        mode='min',
        save_last=True,
        every_n_epochs=1
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=len(args.gpus.split(',')),
        check_val_every_n_epoch=args.val_epochs,
        logger=[tb_logger, wandb_logger],
        callbacks=[checkpoint_callback],
        log_every_n_steps=1
    )
    trainer.fit(vad_model, train_dataloader, val_dataloaders=test_dataloader, ckpt_path=args.resume_from_checkpoint)

if __name__ == '__main__':
    main()