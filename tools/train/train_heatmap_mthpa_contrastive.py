import os
import sys
import argparse
from datetime import datetime

import torch
import torch.distributed
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.lightning.heatmap_mthpa_contrastive import HeatmapMTHPA_Contrastive
from util.file import get_unique_proj_dir_name
from models.dataset.h5_heatmap_contrastive_dataset import get_transform, contrastive_collate_fn, HDF5ContrastiveDataset, SmallHDF5ContrastiveDataset
from models.transformer.mthpa import MTHPA_Base, MTHPA_Large, MTHPA_Small, MTHPA_Tiny

torch.set_float32_matmul_precision('high')

def create_mthpa_model(model_type, num_frames, in_channels):
    """
    Create MTHPA model based on configuration
    """
    if model_type == "tiny":
        return MTHPA_Tiny(num_frames=num_frames, in_channels=in_channels)
    elif model_type == "small":
        return MTHPA_Small(num_frames=num_frames, in_channels=in_channels)
    elif model_type == "base":
        return MTHPA_Base(num_frames=num_frames, in_channels=in_channels)
    elif model_type == "large":
        return MTHPA_Large(num_frames=num_frames, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_model_config(args, mthpa_model, in_channels):
    """
    Print model configuration and hyperparameters
    """
    print("=" * 80)
    print("MTHPA CONTRASTIVE LEARNING CONFIGURATION")
    print("=" * 80)
    
    # Model configuration
    print("\n MODEL CONFIGURATION:")
    print(f"  • Model Type: {args.model_type.upper()}")
    print(f"  • Dataset: {args.dataset}")
    print(f"  • Input Channels: {in_channels}")
    print(f"  • Number of Frames (T): {args.T}")
    print(f"  • Input Shape: (B, {args.T}, {in_channels}, 224, 224)")
    
    # Model parameters
    total_params = sum(p.numel() for p in mthpa_model.parameters())
    trainable_params = sum(p.numel() for p in mthpa_model.parameters() if p.requires_grad)
    print(f"  • Total Parameters: {total_params:,}")
    print(f"  • Trainable Parameters: {trainable_params:,}")
    
    # Model architecture details
    print(f"  • Embedding Dimension: {mthpa_model.embed_dim}")
    print(f"  • Patch Size: {mthpa_model.patch_size}")
    print(f"  • Number of Transformer Layers: {len(mthpa_model.transformer_blocks)}")
    print(f"  • Number of Attention Heads: {mthpa_model.transformer_blocks[0].temporal_attn.attention.num_heads}")
    
    # Training configuration
    print("\n TRAINING CONFIGURATION:")
    print(f"  • Batch Size: {args.batch_size}")
    print(f"  • Learning Rate: {args.lr}")
    print(f"  • Weight Decay: {args.wd}")
    print(f"  • Max Epochs: {args.max_epochs}")
    print(f"  • Validation Every N Epochs: {args.val_epochs}")
    print(f"  • Number of Workers: {args.num_workers}")
    print(f"  • GPUs: {args.gpus}")
    
    # Data configuration
    print("\n DATA CONFIGURATION:")
    print(f"  • Dataset Directory: {args.dataset_dir}")
    print(f"  • Model Save Directory: {args.model_dir}")
    print(f"  • Run Name: {args.run_name if args.run_name else 'Auto-generated'}")
    print(f"  • Resume Checkpoint: {args.resume_checkpoint if args.resume_checkpoint else 'None'}")
    
    # Memory estimation
    input_memory = args.batch_size * args.T * in_channels * 224 * 224 * 4 / (1024**3)  # GB
    print(f"  • Estimated Input Memory: {input_memory:.3f} GB")
    
    print("=" * 80)
    print()

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate VAD model")
    parser.add_argument('--T', type=int, default=16, help="Number of frames")
    parser.add_argument('--model-type', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large'], help='MTHPA model type')
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

    # Check if this is the main process (rank 0) in distributed training
    is_main_process = True
    if torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
    elif 'LOCAL_RANK' in os.environ:
        is_main_process = int(os.environ.get('LOCAL_RANK', 0)) == 0

    if args.resume_checkpoint:
        run_name = args.run_name
        project_dir = args.model_dir
        project_name = f"VAD_{args.dataset}_mthpa-{args.model_type}"
    else:
        if args.run_name == "":
            project_name = f"VAD_{args.dataset}_mthpa-{args.model_type}"
            run_name = f"{args.dataset}_mthpa-{args.model_type}_T{args.T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            project_name = f"VAD_{args.dataset}_mthpa-{args.model_type}"
            run_name = f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        project_dir = get_unique_proj_dir_name(args.model_dir, f"{run_name}")
        # Only create directory on main process
        if is_main_process and not os.path.exists(project_dir):
            os.makedirs(project_dir)
        
        # Synchronize all processes to ensure directory exists before proceeding
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    train_contrastive_h5_file_path = os.path.join(args.dataset_dir, f"train_contrastive_heatmap_f{args.T}.h5")
    test_contrastive_h5_file_path = os.path.join(args.dataset_dir, f"test_contrastive_heatmap_f{args.T}.h5")
    train_contrastive_pairs_csv = os.path.join(args.dataset_dir, f"train_contrastive_heatmap_f{args.T}_contrastive_pairs.csv")
    train_video_list_csv = os.path.join(args.dataset_dir, f"train_contrastive_heatmap_f{args.T}_video_list.csv")
    test_contrastive_pairs_csv = os.path.join(args.dataset_dir, f"test_contrastive_heatmap_f{args.T}_contrastive_pairs.csv")
    test_video_list_csv = os.path.join(args.dataset_dir, f"test_contrastive_heatmap_f{args.T}_video_list.csv")

    # train_contrastive_dataset = SmallHDF5ContrastiveDataset(
    #     train_contrastive_h5_file_path,
    #     train_contrastive_pairs_csv,
    #     train_video_list_csv,
    #     max_length=args.T,
    #     transform=get_transform(),
    #     subset_size=32
    # )
    # test_contrastive_dataset = SmallHDF5ContrastiveDataset(
    #     test_contrastive_h5_file_path,
    #     test_contrastive_pairs_csv,
    #     test_video_list_csv,
    #     max_length=args.T,
    #     transform=get_transform(),
    #     subset_size=32
    # )
    train_contrastive_dataset = HDF5ContrastiveDataset(
        train_contrastive_h5_file_path,
        train_contrastive_pairs_csv,
        train_video_list_csv,
        max_length=args.T,
        transform=get_transform(),
    )
    test_contrastive_dataset = HDF5ContrastiveDataset(
        test_contrastive_h5_file_path,
        test_contrastive_pairs_csv,
        test_video_list_csv,
        max_length=args.T,
        transform=get_transform(),
    )


    train_contrastive_dataloader = DataLoader(train_contrastive_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=contrastive_collate_fn)
    test_contrastive_dataloader = DataLoader(test_contrastive_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=contrastive_collate_fn)

    if args.dataset == "cuhk":
        in_channels = 24
    elif args.dataset == "shanghaitech":
        in_channels = 32
    elif args.dataset == "ubnormal":
        in_channels = 16
    else:
        in_channels = 32

    mthpa_model = create_mthpa_model(args.model_type, args.T, in_channels)

    # Only print config on main process
    if is_main_process:
        print_model_config(args, mthpa_model, in_channels)

    heatmap_mthpa_model = HeatmapMTHPA_Contrastive(
        mthpa_model=mthpa_model,
        wd=args.wd,
        save_dir=project_dir,
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
        log_every_n_steps=1,
    )
    trainer.fit(heatmap_mthpa_model, train_contrastive_dataloader, val_dataloaders=test_contrastive_dataloader, ckpt_path=args.resume_checkpoint)


if __name__ == '__main__':
    main()