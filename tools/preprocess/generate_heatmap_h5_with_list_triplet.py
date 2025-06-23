import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm
import random
import csv


import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm
import random
import csv

def load_npy_files(folder_path, num_frames, in_channels):
    npy_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])
    total_files = len(npy_files)
    interval = max(total_files // num_frames, 1)

    selected_files = []
    frame_numbers = []
    for i in range(0, total_files, interval):
        frame = np.load(npy_files[i])
        if frame.shape[0] != in_channels:
            print(f"Channel mismatch in file {npy_files[i]}: expected {in_channels}, got {frame.shape[0]}")
            continue
        if np.isnan(frame).any():
            print(f"NaN detected in file {npy_files[i]}")
            continue
        selected_files.append(npy_files[i])
        frame_number = int(os.path.splitext(os.path.basename(npy_files[i]))[0].split('_')[-1])
        frame_numbers.append(frame_number)
        if len(selected_files) == num_frames:
            break

    frames = [np.load(f) for f in selected_files]

    return np.array(frames), np.array(frame_numbers)

def create_triplet_h5(output_path, normal_folders, abnormal_folders, num_frames, num_pos, num_neg, csv_path, in_channels):
    video_list = []
    triplet_info = []
    normal_indices = []
    abnormal_indices = []

    with h5py.File(output_path, 'w') as h5f:
        videos_grp = h5f.create_group('videos')

        # Normal videos
        for i, folder in enumerate(tqdm(normal_folders, desc="Adding normal videos")):
            video, _ = load_npy_files(folder, num_frames, in_channels)
            if len(video) < num_frames:
                print(f"Insufficient frames in video from {folder}, skipping...")
                continue
            videos_grp.create_dataset(f'{i}', data=video)
            normal_indices.append(i)
            video_list.append((i, os.path.basename(folder)))

        # Abnormal videos
        abnormal_start_idx = len(normal_folders)
        for i, folder in enumerate(tqdm(abnormal_folders, desc="Adding abnormal videos")):
            video, _ = load_npy_files(folder, num_frames, in_channels)
            if len(video) < num_frames:
                print(f"Insufficient frames in video from {folder}, skipping...")
                continue
            videos_grp.create_dataset(f'{i + abnormal_start_idx}', data=video)
            abnormal_indices.append(i + abnormal_start_idx)
            video_list.append((i + abnormal_start_idx, os.path.basename(folder)))

        # Save indices
        h5f.create_dataset('normal_indices', data=np.array(normal_indices, dtype=np.int32))
        h5f.create_dataset('abnormal_indices', data=np.array(abnormal_indices, dtype=np.int32))

        # Normal triplets
        for anchor_idx in tqdm(normal_indices, desc="Creating normal triplets"):
            positive_indices = random.sample([i for i in normal_indices if i != anchor_idx], min(num_pos, len(normal_indices) - 1))
            negative_indices = random.sample(abnormal_indices, min(num_neg, len(abnormal_indices)))
            triplet_info.extend([(anchor_idx, pos_idx, neg_idx) for pos_idx in positive_indices for neg_idx in negative_indices])

        # Abnormal triplets
        for anchor_idx in tqdm(abnormal_indices, desc="Creating abnormal triplets"):
            positive_indices = random.sample([i for i in abnormal_indices if i != anchor_idx], min(num_pos, len(abnormal_indices) - 1))
            negative_indices = random.sample(normal_indices, min(num_neg, len(normal_indices)))
            triplet_info.extend([(anchor_idx, pos_idx, neg_idx) for pos_idx in positive_indices for neg_idx in negative_indices])

        triplet_info_arr = np.array(triplet_info, dtype=np.int32)
        h5f.create_dataset('triplet_info', data=triplet_info_arr)

    print(f"Saved triplet index data to {output_path}")

    # Save video list to CSV
    with open(os.path.join(csv_path, f'video_list_f{num_frames}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Video'])
        writer.writerows(video_list)

    # Save triplet info to CSV
    with open(os.path.join(csv_path, f'triplet_info_f{num_frames}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Anchor', 'Positive', 'Negative'])
        writer.writerows(triplet_info)

def main():
    parser = argparse.ArgumentParser(description="Process video frames to generate HDF5 files with uniform frame sampling.")
    parser.add_argument('--list-dir', type=str, required=True, help='Path to the directory containing train.list and test.list.')
    parser.add_argument('--dataset', choices=["cuhk", "shanghaitech"], required=True, help="Type of the dataset")
    parser.add_argument('--npy-dir', type=str, required=True, help='Path to the directory containing video frame folders (training/frames and testing/frames).')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory where HDF5 files will be saved.')
    parser.add_argument('--num-pos', type=int, default=3, help='Number of positive samples per anchor(cuhk pos: 2~3 / shanghaitech pos: 5~7)')
    parser.add_argument('--num-neg', type=int, default=5, help='Number of negative samples per anchor(cuhk neg: 3~5 / shanghaitech neg: 10~15)')
    args = parser.parse_args()

    train_list_file = os.path.join(args.list_dir, "train.list")
    test_list_file = os.path.join(args.list_dir, "test.list")

    with open(train_list_file, 'r') as file:
        train_video_list = file.read().strip().split('\n')

    with open(test_list_file, 'r') as file:
        test_video_list = file.read().strip().split('\n')

    train_video_folders = []
    test_video_folders = []

    if args.dataset == "cuhk":
        in_channels = 24
    elif args.dataset == "shanghaitech":
        in_channels = 32
    else:
        in_channels = 32

    for video_name in train_video_list:
        if args.dataset == "cuhk":
            train_video_folder = os.path.join(args.npy_dir, video_name)
            if os.path.isdir(train_video_folder):
                train_video_folders.append(train_video_folder)
        else:
            train_video_folder = os.path.join(args.npy_dir, "train", video_name)
            if os.path.isdir(train_video_folder):
                train_video_folders.append(train_video_folder)
            test_video_folder = os.path.join(args.npy_dir, "test", video_name)
            if os.path.isdir(test_video_folder):
                train_video_folders.append(test_video_folder)

    for video_name in test_video_list:
        if args.dataset == "cuhk":
            test_video_folder = os.path.join(args.npy_dir, video_name)
            if os.path.isdir(test_video_folder):
                test_video_folders.append(test_video_folder)
        else:
            test_video_folder = os.path.join(args.npy_dir, "test", video_name)
            if os.path.isdir(test_video_folder):
                test_video_folders.append(test_video_folder)
            train_video_folder = os.path.join(args.npy_dir, "train", video_name)
            if os.path.isdir(train_video_folder):
                test_video_folders.append(train_video_folder)

    train_video_folders_normal = [folder for folder in train_video_folders if 'train' in folder]
    train_video_folders_abnormal = [folder for folder in train_video_folders if 'test' in folder]
    test_video_folders_normal = [folder for folder in test_video_folders if 'train' in folder]
    test_video_folders_abnormal = [folder for folder in test_video_folders if 'test' in folder]

    frame_counts = [16, 32, 64, 128]
    for count in frame_counts:
        print(f"Starting to generate HDF5 files for {count} frames (test: {len(test_video_folders_normal) + len(test_video_folders_abnormal)} / train: {len(train_video_folders_normal) + len(train_video_folders_abnormal)})")

        train_triplet_h5_path = os.path.join(args.output_dir, f'train_triplet_heatmap_f{count}.h5')
        create_triplet_h5(train_triplet_h5_path, train_video_folders_normal, train_video_folders_abnormal, count, args.num_pos, args.num_neg, args.output_dir, in_channels)

        test_triplet_h5_path = os.path.join(args.output_dir, f'test_triplet_heatmap_f{count}.h5')
        create_triplet_h5(test_triplet_h5_path, test_video_folders_normal, test_video_folders_abnormal, count, args.num_pos, args.num_neg, args.output_dir, in_channels)


if __name__ == "__main__":
    main()
