import os
import sys
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.file import get_unique_proj_dir_name
from models.dataset.h5_heatmap_triplet_dataset import get_transform, collate_fn, HDF5AnchorDataset
from models.c3d.heatmap_i3d_4c import HeatmapI3D
from models.c3d.heatmap_a3d_4c import HeatmapA3D
from models.c3d.heatmap_resnet3d_4c import HeatmapResNetD4C
from models.vad.heatmap_3d_triplet import Heatmap3D_Triplet
from models.ae.ae_trainer import AETrainer
from models.ae.autoencoder import Autoencoder
from models.ae.conv_ae import ConvAE
from models.ae.lstem_ae import LSTMAE
from models.mem_ae import MemAE
from models.dataset.feature_dataset import FeatureDataset
from util.model import load_model


# torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate VAD model")
    parser.add_argument('--T', type=int, default=16, help="Number of frames")
    parser.add_argument('--backbone', type=str, default="i3d", choices=['resnet3d', 'i3d', 'a3d'], required=True, help="Backbone model")
    parser.add_argument('--backbone-checkpoint', type=str, help="Path to a checkpoint to resume from", default=None)
    parser.add_argument('--resnet-depth', type=int, choices=[18, 34, 50, 101, 152], help="ResNet depth (only for resnet3d and a3d)")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size")
    parser.add_argument('--max-epochs', type=int, default=200, help="Maximum number of epochs")
    parser.add_argument('--val-epochs', type=int, default=5, help="Validation epochs")
    parser.add_argument('--lr', type=float, default=0.000001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.000005, help="Weight decay")
    parser.add_argument('--dataset', choices=["cuhk", "shanghaitech", "ubnormal"], required=True, help="Type of the dataset")
    parser.add_argument('--dataset-dir', type=str, required=True, help="Directory path to the training HDF5 file")
    parser.add_argument('--model-dir', type=str, required=True, help="Directory path to save model checkpoints")
    parser.add_argument('--resume-checkpoint', type=str, help="Path to a checkpoint to resume from", default=None)
    parser.add_argument('--run-name', type=str, default="", help="Project name")
    parser.add_argument('--gpus', type=str, default="0", help="Comma-separated list of GPU ids to use")
    parser.add_argument('--num-workers', type=int, default=2, help="Number of data loading workers")
    parser.add_argument('--run-id', type=str, default=None, help="WandB run ID to resume")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.resume_checkpoint:
        run_name = args.run_name
        project_dir = args.model_dir
        project_name = f"VAD_{args.dataset}_ae"
    else:
        if args.run_name == "":
            project_name = f"VAD_{args.dataset}_ae"
            if args.backbone in ['resnet3d', 'a3d']:
                run_name = f"{args.dataset}_{args.backbone}_{args.resnet_depth}_T{args.T}_ae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                run_name = f"{args.dataset}_{args.backbone}_T{args.T}_ae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            project_name = f"VAD_{args.dataset}_ae"
            run_name = f"{args.run_name}_T{args.T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        project_dir = get_unique_proj_dir_name(args.model_dir, f"{run_name}")
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

    if args.dataset == "cuhk":
        in_channels = 24
        encoding_dim = 128
    elif args.dataset == "shanghaitech":
        in_channels = 32
        encoding_dim = 256
    elif args.dataset == "ubnormal":
        in_channels = 16
        encoding_dim = 64
    else:
        in_channels = 32
        encoding_dim = 128

    train_h5_file_path = os.path.join(args.dataset_dir, f"train_triplet_heatmap_f{args.T}.h5")
    test_h5_file_path = os.path.join(args.dataset_dir, f"test_triplet_heatmap_f{args.T}.h5")

    train_anchor_dataset = HDF5AnchorDataset(train_h5_file_path, max_length=args.T, depth=in_channels, transform=get_transform(), log=False)
    test_anchor_dataset = HDF5AnchorDataset(test_h5_file_path, max_length=args.T, depth=in_channels, transform=get_transform(), log=False)

    train_anchor_dataloader = DataLoader(train_anchor_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_anchor_dataloader = DataLoader(test_anchor_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.backbone == 'resnet3d':
        heatmap_c3d = HeatmapResNetD4C(depth=args.resnet_depth, in_channels=in_channels)
        if args.resnet_depth in [18, 34]:
            input_dim = 512
        else:
            input_dim = 2048
    elif args.backbone == 'a3d':
        heatmap_c3d = HeatmapA3D(depth=args.resnet_depth, in_channels=in_channels)
        if args.resnet_depth in [18, 34]:
            input_dim = 512
        else:
            input_dim = 2048
    else:
        heatmap_c3d = HeatmapI3D(in_channels=in_channels)
        input_dim = 1024

    if args.backbone_checkpoint:
        heatmap_c3d = load_model(args.backbone_checkpoint, in_channels)

    # autoencoder = ConvAE(input_dim=input_dim)
    autoencoder = LSTMAE(input_dim=input_dim)
    # autoencoder = MemAE(input_dim=input_dim, mem_dim=500)

    if args.resume_checkpoint and not args.resume_checkpoint.endswith('.pt'):
        ae_model = AETrainer(
            feature_extractor=heatmap_c3d,
            autoencoder=autoencoder,
            lr=args.lr,
            wd=args.wd,
            save_dir=project_dir
        )
        ae_model = ae_model.load_from_checkpoint(args.resume_checkpoint)
    else:
        ae_model = AETrainer(
            feature_extractor=heatmap_c3d,
            autoencoder=autoencoder,
            lr=args.lr,
            wd=args.wd,
            save_dir=project_dir
        )

    wandb_logger = WandbLogger(project=project_name, name=run_name, id=args.run_id, resume="allow")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=project_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        every_n_epochs=1
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=len(args.gpus.split(',')),
        strategy=DDPStrategy(find_unused_parameters=True),
        check_val_every_n_epoch=args.val_epochs,
        logger=[wandb_logger],
        callbacks=[checkpoint_callback],
        log_every_n_steps=1
    )
    trainer.fit(ae_model, train_anchor_dataloader, val_dataloaders=test_anchor_dataloader, ckpt_path=args.resume_checkpoint)

if __name__ == '__main__':
    main()