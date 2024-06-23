import os
import sys
import argparse
from datetime import datetime

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.file import get_unique_proj_dir_name
from models.dataset.h5_heatmap_dataset import get_transform, HDF5TripletDataset
from models.c3d.heatmap_c3d_4c import HeatmapC3D4C
from models.vad.heatmap_3d_triplet import ResNet3D_Triplet


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate VAD model")
    parser.add_argument('--T', type=int, default=16, help="Number of frames")
    parser.add_argument('--c3d-depth', type=int, choices=[18, 34, 50, 101, 152], default=18, help="ResNet depth")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size")
    parser.add_argument('--max-epochs', type=int, default=500, help="Maximum number of epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.000005, help="Weight decay")
    parser.add_argument('--dataset', choices=["cuhk", "shanghaitech", "ubnormal"], required=True, help="Type of the dataset")
    parser.add_argument('--dataset-dir', type=str, required=True, help="Directory path to the training HDF5 file")
    parser.add_argument('--model-dir', type=str, required=True, help="Directory path to save model checkpoints")
    parser.add_argument('--resume_from_checkpoint', type=str, help="Path to a checkpoint to resume from", default=None)
    parser.add_argument('--run-name', type=str, default="", help="Project name")
    parser.add_argument('--gpus', type=str, default="0", help="Comma-separated list of GPU ids to use")
    parser.add_argument('--num-workers', type=int, default=2, help="Number of data loading workers")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    project_name = f"VAD_ResNet{args.c3d_depth}_{args.dataset}"
    if args.run_name == "":
        run_name = f"{args.dataset}_32x224x224_T{args.T}_ResNet{args.c3d_depth}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_name = f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project_dir = get_unique_proj_dir_name(args.model_dir, f"{args.dataset}_heatmap_f{args.T}")
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

        train_triplet_h5_file_path = os.path.join(args.dataset_dir,
                                                  f"{args.dataset}_train_triplet_heatmap_f{args.T}.h5")
        test_triplet_h5_file_path = os.path.join(args.dataset_dir, f"{args.dataset}_test_triplet_heatmap_f{args.T}.h5")

        train_triplet_dataset = HDF5TripletDataset(train_triplet_h5_file_path, transform=get_transform())
        test_triplet_dataset = HDF5TripletDataset(test_triplet_h5_file_path, transform=get_transform())

        train_triplet_dataloader = DataLoader(train_triplet_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers)
        test_triplet_dataloader = DataLoader(test_triplet_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)

        heatmap_c3d = HeatmapC3D4C(depth=args.c3d_depth, in_channels=32)

        resnet3d_model = ResNet3D_Triplet(
            heatmap_c3d=heatmap_c3d,
            lr=args.lr,
            wd=args.wd
        )

        tb_logger = TensorBoardLogger("tb_logs", name=project_name)
        wandb_logger = WandbLogger(project=project_name, name=run_name)
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
            check_val_every_n_epoch=args.val_epochs,
            logger=[tb_logger, wandb_logger],
            callbacks=[checkpoint_callback],
            log_every_n_steps=1
        )
        trainer.fit(resnet3d_model, train_triplet_dataloader, val_dataloaders=test_triplet_dataloader,
                    ckpt_path=args.resume_from_checkpoint)

    if __name__ == '__main__':
        main()