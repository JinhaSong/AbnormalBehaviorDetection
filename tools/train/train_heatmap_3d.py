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
from models.dataset.h5_heatmap_triplet_dataset import get_transform, triplet_collate_fn, HDF5TripletDataset, SmallHDF5TripletDataset
from models.c3d.heatmap_i3d_4c import HeatmapI3D
from models.c3d.heatmap_a3d_4c import HeatmapA3D
from models.c3d.heatmap_resnet3d_4c import HeatmapResNetD4C
from models.vivit.heatmap_vivit import HeatmapViViT
from models.lightning.heatmap_3d_triplet import Heatmap3D_Triplet


torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate VAD model")
    parser.add_argument('--T', type=int, default=16, help="Number of frames")
    parser.add_argument('--backbone', type=str, default="i3d", choices=['resnet3d', 'i3d', 'a3d', 'vivit'], required=True, help="Backbone model")
    parser.add_argument('--resnet-depth', type=int, choices=[18, 34, 50, 101, 152], help="ResNet depth (only for resnet3d and a3d)")
    parser.add_argument('--embed-dim', type=int, choices=[384, 768, 1024], help="Embedding dimension (only for vivit)")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size")
    parser.add_argument('--max-epochs', type=int, default=200, help="Maximum number of epochs")
    parser.add_argument('--val-epochs', type=int, default=5, help="Validation epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
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
        if args.backbone in ['resnet3d', 'a3d']:
            project_name = f"VAD_{args.dataset}_{args.backbone}_{args.resnet_depth}"
        else:
            project_name = f"VAD_{args.dataset}_{args.backbone}"
    else:
        if args.run_name == "":
            if args.backbone in ['resnet3d', 'a3d']:
                project_name = f"VAD_{args.dataset}_{args.backbone}_{args.resnet_depth}"
                if args.run_name == "":
                    run_name = f"{args.dataset}_{args.backbone}_{args.resnet_depth}_T{args.T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                else:
                    run_name = f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                project_name = f"VAD_{args.dataset}_{args.backbone}"
                run_name = f"{args.dataset}_{args.backbone}_T{args.T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            project_name = f"VAD_{args.dataset}_{args.backbone}"
            run_name = f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        project_dir = get_unique_proj_dir_name(args.model_dir, f"{run_name}")
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

    train_triplet_h5_file_path = os.path.join(args.dataset_dir, f"train_triplet_heatmap_f{args.T}.h5")
    test_triplet_h5_file_path = os.path.join(args.dataset_dir, f"test_triplet_heatmap_f{args.T}.h5")

    train_triplet_dataset = HDF5TripletDataset(train_triplet_h5_file_path, max_length=args.T, transform=get_transform(), log=False)
    test_triplet_dataset = HDF5TripletDataset(test_triplet_h5_file_path, max_length=args.T, transform=get_transform(), log=False)
    # train_triplet_dataset = SmallHDF5TripletDataset(train_triplet_h5_file_path, max_length=args.T, transform=get_transform(), subset_size=32)
    # test_triplet_dataset = SmallHDF5TripletDataset(test_triplet_h5_file_path, max_length=args.T, transform=get_transform(), subset_size=32)

    train_triplet_dataloader = DataLoader(train_triplet_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=triplet_collate_fn)
    test_triplet_dataloader = DataLoader(test_triplet_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=triplet_collate_fn)

    if args.dataset == "cuhk":
        in_channels = 24
    elif args.dataset == "shanghaitech":
        in_channels = 32
    elif args.dataset == "ubnormal":
        in_channels = 16
    else:
        in_channels = 32

    if args.backbone == 'resnet3d':
        heatmap_c3d = HeatmapResNetD4C(depth=args.c3d_depth, in_channels=in_channels)
    elif args.backbone == 'a3d':
        heatmap_c3d = HeatmapA3D(depth=args.c3d_depth, in_channels=in_channels)
    elif args.backbone == 'vivit':
        heatmap_c3d = HeatmapViViT(num_frames=args.T, in_channels=in_channels, embed_dim=args.embed_dim)
    else:
        heatmap_c3d = HeatmapI3D(in_channels=in_channels)


    heatmap3d_model = Heatmap3D_Triplet(
        heatmap_c3d=heatmap_c3d,
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
        strategy=DDPStrategy(find_unused_parameters=False),
        check_val_every_n_epoch=args.val_epochs,
        logger=[wandb_logger],
        callbacks=[checkpoint_callback],
        log_every_n_steps=1
    )
    trainer.fit(heatmap3d_model, train_triplet_dataloader, val_dataloaders=test_triplet_dataloader, ckpt_path=args.resume_checkpoint)


if __name__ == '__main__':
    main()