import os
import sys
import argparse
from datetime import datetime
import gc

import torch
import torch.distributed
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.strategies import DDPStrategy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.lightning.heatmap_mthpa_contrastive import HeatmapMTHPA_Contrastive
from util.file import get_unique_proj_dir_name
from models.dataset.h5_heatmap_contrastive_dataset import get_transform, contrastive_collate_fn, HDF5ContrastiveDataset, SmallHDF5ContrastiveDataset
from models.transformer.mthpa import MTHPA_Base, MTHPA_Large, MTHPA_Small, MTHPA_Tiny

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.set_float32_matmul_precision('high')


class MemoryMonitorCallback(Callback):
    """Callback to monitor GPU memory usage"""
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_interval == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            # print(f"[Step {trainer.global_step}] GPU Memory - "
            #       f"Allocated: {allocated:.2f}GB, "
            #       f"Reserved: {reserved:.2f}GB, "
            #       f"Peak: {max_allocated:.2f}GB")
            
            # Clear cache periodically
            if batch_idx % (self.log_interval * 5) == 0:
                torch.cuda.empty_cache()
                gc.collect()


class GradientAccumulationScheduler(Callback):
    """Callback to schedule gradient accumulation"""
    def __init__(self, scheduling_dict):
        self.scheduling_dict = scheduling_dict

    def on_epoch_start(self, trainer, pl_module):
        for epoch in sorted(self.scheduling_dict.keys()):
            if trainer.current_epoch >= epoch:
                trainer.accumulate_grad_batches = self.scheduling_dict[epoch]
        print(f"Gradient accumulation steps: {trainer.accumulate_grad_batches}")


def create_mthpa_model(model_type, num_frames, in_channels):
    """Create MTHPA model based on configuration"""
    model_dict = {
        "tiny": MTHPA_Tiny,
        "small": MTHPA_Small,
        "base": MTHPA_Base,
        "large": MTHPA_Large
    }
    
    if model_type not in model_dict:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_dict[model_type](num_frames=num_frames, in_channels=in_channels)


def estimate_memory_usage(args, model, in_channels):
    """Estimate memory usage for the configuration"""
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    param_memory = total_params * 4 / (1024**3)  # GB (float32)
    
    # Input tensor memory per batch
    input_memory = (args.batch_size * args.T * in_channels * 224 * 224 * 4) / (1024**3)
    
    # Activation memory (rough estimate based on model size)
    if args.model_type == "tiny":
        activation_multiplier = 8
    elif args.model_type == "small":
        activation_multiplier = 10
    elif args.model_type == "base":
        activation_multiplier = 12
    else:  # large
        activation_multiplier = 15
    
    activation_memory = input_memory * activation_multiplier
    
    # Gradient memory
    grad_memory = param_memory * 2  # Gradients + optimizer states
    
    total_memory = param_memory + input_memory + activation_memory + grad_memory
    
    return {
        'param_memory': param_memory,
        'input_memory': input_memory,
        'activation_memory': activation_memory,
        'grad_memory': grad_memory,
        'total_memory': total_memory
    }


def get_recommended_batch_size(model_type, num_frames, gpu_memory_gb):
    """Get recommended batch size based on GPU memory"""
    # Conservative batch size recommendations
    if num_frames <= 16:
        if gpu_memory_gb >= 80:  # A100 80GB
            batch_sizes = {"tiny": 32, "small": 16, "base": 8, "large": 4}
        elif gpu_memory_gb >= 48:  # A6000 48GB
            batch_sizes = {"tiny": 16, "small": 8, "base": 4, "large": 2}
        elif gpu_memory_gb >= 32:  # V100 32GB
            batch_sizes = {"tiny": 8, "small": 4, "base": 2, "large": 1}
        else:  # 24GB or less
            batch_sizes = {"tiny": 4, "small": 2, "base": 1, "large": 1}
    elif num_frames <= 32:
        if gpu_memory_gb >= 80:
            batch_sizes = {"tiny": 16, "small": 8, "base": 4, "large": 2}
        elif gpu_memory_gb >= 48:
            batch_sizes = {"tiny": 8, "small": 4, "base": 2, "large": 1}
        else:
            batch_sizes = {"tiny": 4, "small": 2, "base": 1, "large": 1}
    else:  # num_frames > 32
        if gpu_memory_gb >= 80:
            batch_sizes = {"tiny": 8, "small": 4, "base": 2, "large": 1}
        elif gpu_memory_gb >= 48:
            batch_sizes = {"tiny": 4, "small": 2, "base": 1, "large": 1}
        else:
            batch_sizes = {"tiny": 2, "small": 1, "base": 1, "large": 1}
    
    return batch_sizes.get(model_type, 1)


def print_model_config(args, mthpa_model, in_channels, memory_estimate=None):
    """Print model configuration and hyperparameters"""
    print("=" * 80)
    print("MTHPA CONTRASTIVE LEARNING CONFIGURATION")
    print("=" * 80)
    
    # Model configuration
    print("\n📊 MODEL CONFIGURATION:")
    print(f"  • Model Type: {args.model_type.upper()}")
    print(f"  • Dataset: {args.dataset}")
    print(f"  • Input Channels: {in_channels}")
    print(f"  • Number of Frames (T): {args.T}")
    print(f"  • Input Shape: (B, {args.T}, {in_channels}, 224, 224)")
    
    # Model parameters
    total_params = sum(p.numel() for p in mthpa_model.parameters())
    trainable_params = sum(p.numel() for p in mthpa_model.parameters() if p.requires_grad)
    print(f"  • Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  • Trainable Parameters: {trainable_params:,}")
    
    # Model architecture details
    print(f"  • Embedding Dimension: {mthpa_model.embed_dim}")
    print(f"  • Patch Size: {mthpa_model.patch_size}")
    print(f"  • Number of Transformer Layers: {len(mthpa_model.transformer_blocks)}")
    print(f"  • Number of Attention Heads: {mthpa_model.transformer_blocks[0].temporal_attn.attention.num_heads}")
    
    # Training configuration
    print("\n🚀 TRAINING CONFIGURATION:")
    print(f"  • Batch Size: {args.batch_size}")
    if hasattr(args, 'gradient_accumulation') and args.gradient_accumulation > 1:
        print(f"  • Gradient Accumulation Steps: {args.gradient_accumulation}")
        print(f"  • Effective Batch Size: {args.batch_size * args.gradient_accumulation}")
    print(f"  • Learning Rate: {args.lr}")
    print(f"  • Weight Decay: {args.wd}")
    print(f"  • Max Epochs: {args.max_epochs}")
    print(f"  • Validation Every N Epochs: {args.val_epochs}")
    print(f"  • Number of Workers: {args.num_workers}")
    print(f"  • GPUs: {args.gpus}")
    print(f"  • Mixed Precision (FP16): {getattr(args, 'use_fp16', False)}")
    
    # Data configuration
    print("\n📁 DATA CONFIGURATION:")
    print(f"  • Dataset Directory: {args.dataset_dir}")
    print(f"  • Model Save Directory: {args.model_dir}")
    print(f"  • Run Name: {args.run_name if args.run_name else 'Auto-generated'}")
    print(f"  • Resume Checkpoint: {args.resume_checkpoint if args.resume_checkpoint else 'None'}")
    
    # Memory estimation
    if memory_estimate:
        print("\n💾 MEMORY ESTIMATION:")
        print(f"  • Model Parameters: {memory_estimate['param_memory']:.2f} GB")
        print(f"  • Input Tensor: {memory_estimate['input_memory']:.3f} GB")
        print(f"  • Activations (est.): {memory_estimate['activation_memory']:.2f} GB")
        print(f"  • Gradients & Optimizer: {memory_estimate['grad_memory']:.2f} GB")
        print(f"  • Total Estimated: {memory_estimate['total_memory']:.2f} GB")
    
    # GPU information
    if torch.cuda.is_available():
        print("\n🖥️  GPU INFORMATION:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  • GPU {i}: {props.name}")
            print(f"    - Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    - Compute Capability: {props.major}.{props.minor}")
    
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate VAD model")
    
    # Model configuration
    parser.add_argument('--T', type=int, default=16, help="Number of frames")
    parser.add_argument('--model-type', type=str, default='tiny', 
                        choices=['tiny', 'small', 'base', 'large'], 
                        help='MTHPA model type')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size")
    parser.add_argument('--gradient-accumulation', type=int, default=1, 
                        help="Gradient accumulation steps (for memory efficiency)")
    parser.add_argument('--max-epochs', type=int, default=200, help="Maximum number of epochs")
    parser.add_argument('--val-epochs', type=int, default=5, help="Validation epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.000005, help="Weight decay")
    parser.add_argument('--use-fp16', action='store_true', 
                        help="Use mixed precision training (not recommended for pose data)")
    
    # Dataset configuration
    parser.add_argument('--dataset', choices=["cuhk", "shanghaitech", "ubnormal"], 
                        required=True, help="Type of the dataset")
    parser.add_argument('--dataset-dir', type=str, required=True, 
                        help="Directory path to the training HDF5 file")
    parser.add_argument('--model-dir', type=str, required=True, 
                        help="Directory path to save model checkpoints")
    
    # Hardware configuration
    parser.add_argument('--gpus', type=str, default="0", 
                        help="Comma-separated list of GPU ids to use")
    parser.add_argument('--num-workers', type=int, default=2, 
                        help="Number of data loading workers")
    
    # Resume and logging
    parser.add_argument('--resume-checkpoint', type=str, help="Path to a checkpoint to resume from", default=None)
    parser.add_argument('--run-name', type=str, default="", help="Project name")
    parser.add_argument('--run-id', type=str, default=None, help="WandB run ID to resume")
    
    # Memory optimization
    parser.add_argument('--auto-batch-size', action='store_true', 
                        help="Automatically determine batch size based on GPU memory")
    parser.add_argument('--use-small-dataset', action='store_true', 
                        help="Use small subset for testing")

    args = parser.parse_args()

    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Check if this is the main process (rank 0) in distributed training
    is_main_process = True
    if torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
    elif 'LOCAL_RANK' in os.environ:
        is_main_process = int(os.environ.get('LOCAL_RANK', 0)) == 0

    # Auto-detect batch size if requested
    if args.auto_batch_size and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        args.batch_size = get_recommended_batch_size(args.model_type, args.T, gpu_memory)
        if is_main_process:
            print(f"Auto-detected batch size: {args.batch_size} for {gpu_memory:.1f}GB GPU")

    # Setup directories
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

    # Setup data paths
    train_contrastive_h5_file_path = os.path.join(args.dataset_dir, f"train_contrastive_heatmap_f{args.T}.h5")
    test_contrastive_h5_file_path = os.path.join(args.dataset_dir, f"test_contrastive_heatmap_f{args.T}.h5")
    train_contrastive_pairs_csv = os.path.join(args.dataset_dir, f"train_contrastive_heatmap_f{args.T}_contrastive_pairs.csv")
    train_video_list_csv = os.path.join(args.dataset_dir, f"train_contrastive_heatmap_f{args.T}_video_list.csv")
    test_contrastive_pairs_csv = os.path.join(args.dataset_dir, f"test_contrastive_heatmap_f{args.T}_contrastive_pairs.csv")
    test_video_list_csv = os.path.join(args.dataset_dir, f"test_contrastive_heatmap_f{args.T}_video_list.csv")

    # Determine number of channels based on dataset
    channel_map = {
        "cuhk": 24,
        "shanghaitech": 32,
        "ubnormal": 16
    }
    in_channels = channel_map.get(args.dataset, 32)

    # Create datasets
    if args.use_small_dataset:
        # Use small dataset for testing
        train_contrastive_dataset = SmallHDF5ContrastiveDataset(
            train_contrastive_h5_file_path,
            train_contrastive_pairs_csv,
            train_video_list_csv,
            max_length=args.T,
            transform=get_transform(),
            subset_size=32,
            depth=in_channels
        )
        test_contrastive_dataset = SmallHDF5ContrastiveDataset(
            test_contrastive_h5_file_path,
            test_contrastive_pairs_csv,
            test_video_list_csv,
            max_length=args.T,
            transform=get_transform(),
            subset_size=32,
            depth=in_channels
        )
    else:
        # Use full dataset
        train_contrastive_dataset = HDF5ContrastiveDataset(
            train_contrastive_h5_file_path,
            train_contrastive_pairs_csv,
            train_video_list_csv,
            max_length=args.T,
            transform=get_transform(),
            depth=in_channels
        )
        test_contrastive_dataset = HDF5ContrastiveDataset(
            test_contrastive_h5_file_path,
            test_contrastive_pairs_csv,
            test_video_list_csv,
            max_length=args.T,
            transform=get_transform(),
            depth=in_channels
        )

    # Create dataloaders with memory-efficient settings
    train_contrastive_dataloader = DataLoader(
        train_contrastive_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        collate_fn=contrastive_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True  # Important for stable batch norm
    )
    
    test_contrastive_dataloader = DataLoader(
        test_contrastive_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=contrastive_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )

    # Create model
    mthpa_model = create_mthpa_model(args.model_type, args.T, in_channels)

    # Estimate memory usage
    memory_estimate = estimate_memory_usage(args, mthpa_model, in_channels)
    
    # Only print config on main process
    if is_main_process:
        print_model_config(args, mthpa_model, in_channels, memory_estimate)
        
        # Check if configuration is feasible
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if memory_estimate['total_memory'] > gpu_memory * 0.9:  # 90% threshold
                print("\n⚠️  WARNING: Estimated memory usage exceeds available GPU memory!")
                print(f"   Estimated: {memory_estimate['total_memory']:.2f}GB, Available: {gpu_memory:.2f}GB")
                print(f"   Consider reducing batch size or using gradient accumulation.")

    # Create Lightning module
    heatmap_mthpa_model = HeatmapMTHPA_Contrastive(
        mthpa_model=mthpa_model,
        lr=args.lr,
        wd=args.wd,
        save_dir=project_dir,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_mixed_precision=args.use_fp16,
        max_stored_features=1000  # Limit stored features for visualization
    )

    # Setup logging
    wandb_logger = WandbLogger(project=project_name, name=run_name, id=args.run_id, resume="allow")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=project_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        every_n_epochs=1
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    memory_monitor = MemoryMonitorCallback(log_interval=100)
    
    # Create list of callbacks
    callbacks = [checkpoint_callback, lr_monitor, memory_monitor]
    
    # Add gradient accumulation scheduler if needed
    if args.gradient_accumulation > 1:
        # You can modify this schedule as needed
        grad_acc_schedule = {
            0: args.gradient_accumulation,
            50: max(1, args.gradient_accumulation // 2),  # Reduce after 50 epochs
            100: 1  # No accumulation after 100 epochs
        }
        callbacks.append(GradientAccumulationScheduler(grad_acc_schedule))

    # Create trainer with optimized settings
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=len(args.gpus.split(',')),
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=16 if args.use_fp16 else 32,
        accumulate_grad_batches=args.gradient_accumulation,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.val_epochs,
        logger=[wandb_logger],
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        num_sanity_val_steps=2,
        limit_val_batches=1.0,  # Use full validation set
        # Debugging
        detect_anomaly=False,  # Set to True for debugging
        profiler=None,  # Set to "simple" or "advanced" for profiling
    )
    
    # Train model
    trainer.fit(
        heatmap_mthpa_model, 
        train_contrastive_dataloader, 
        val_dataloaders=test_contrastive_dataloader, 
        ckpt_path=args.resume_checkpoint
    )


if __name__ == '__main__':
    main()