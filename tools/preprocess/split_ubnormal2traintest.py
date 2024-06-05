import os
import random
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path

def sample_directories_from_dir(frames_dir, train_ratio=0.7):
    all_dirs = [d for d in Path(frames_dir).iterdir() if d.is_dir()]
    random.shuffle(all_dirs)
    split_index = int(len(all_dirs) * train_ratio)

    train_dirs = all_dirs[:split_index]
    test_dirs = all_dirs[split_index:]

    return train_dirs, test_dirs

def move_directories(dir_list, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for dir_path in tqdm(dir_list, desc=f"Moving directories to {target_dir}"):
        target_path = os.path.join(target_dir, os.path.basename(dir_path))
        shutil.move(str(dir_path), target_path)

def sample_and_move_directories(frames_dir, output_dir, train_ratio=0.7):
    train_dirs, test_dirs = sample_directories_from_dir(frames_dir, train_ratio)

    train_output_dir = os.path.join(output_dir, "frames", "train")
    test_output_dir = os.path.join(output_dir, "frames", "test")

    move_directories(train_dirs, train_output_dir)
    move_directories(test_dirs, test_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train-ratio", default=0.7, help="Ratio of training and testing video")
    parser.add_argument("--frames-dir", required=True, help="Path to UBnormal dataset directory")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")

    opt = parser.parse_known_args()[0]

    train_ratio = float(opt.train_ratio)
    frames_dir = opt.frames_dir
    output_dir = opt.output_dir

    sample_and_move_directories(frames_dir, output_dir, train_ratio=train_ratio)