import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm
import csv
import random

def resize_frame(frame, target_size=(32, 224, 224)):
    return np.resize(frame, target_size)

def load_npy_files(folder_path, num_frames):
    npy_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])
    total_files = len(npy_files)
    interval = max(total_files // num_frames, 1)

    selected_files = []
    frame_numbers = []
    for i in range(0, total_files, interval):
        selected_files.append(npy_files[i])
        frame_number = int(os.path.splitext(os.path.basename(npy_files[i]))[0].split('_')[-1])
        frame_numbers.append(frame_number)
        if len(selected_files) == num_frames:
            break

    frames = [resize_frame(np.load(f)) for f in selected_files]
    return np.array(frames), np.array(frame_numbers)

def save_h5_file(output_path, video_folders, num_frames):
    h5f = h5py.File(output_path, 'w')

    for video_folder in tqdm(video_folders, desc=f"Processing {num_frames} frames"):
        if not os.path.exists(video_folder):
            print(f"Video folder {video_folder} does not exist. Skipping.")
            continue

        frames, frame_numbers = load_npy_files(video_folder, num_frames)

        group_name = os.path.join(os.path.basename(os.path.dirname(video_folder)), os.path.basename(video_folder))
        group = h5f.create_group(group_name)
        group.create_dataset('video_data', data=frames)
        group.create_dataset('frame_numbers', data=frame_numbers)

        if 'abnormal' in video_folder:
            label = 1  # anomaly
        else:
            label = 0  # normal
        group.create_dataset('label', data=label)

    h5f.close()
    print(f"Saved {num_frames} frames per video to {output_path}")

def create_triplet_h5(output_path, normal_folders, abnormal_folders, num_frames, csv_path, num_positive, num_negative):
    h5f = h5py.File(output_path, 'w')
    csv_data = []

    normal_videos = [load_npy_files(folder, num_frames)[0] for folder in normal_folders]
    abnormal_videos = [load_npy_files(folder, num_frames)[0] for folder in abnormal_folders]

    triplet_index = 0

    # Normal triplets
    for anchor_idx, anchor in enumerate(tqdm(normal_videos, desc="Creating normal triplets")):
        positive_indices = random.sample([i for i in range(len(normal_videos)) if i != anchor_idx], min(num_positive, len(normal_videos)-1))
        negative_indices = random.sample(range(len(abnormal_videos)), min(num_negative, len(abnormal_videos)))

        for positive_idx in positive_indices:
            for negative_idx in negative_indices:
                triplet_group = h5f.create_group(f'triplet_{triplet_index}')
                triplet_group.create_dataset('anchor', data=anchor)
                triplet_group.create_dataset('positive', data=normal_videos[positive_idx])
                triplet_group.create_dataset('negative', data=abnormal_videos[negative_idx])

                anchor_name = normal_folders[anchor_idx]
                positive_name = normal_folders[positive_idx]
                negative_name = abnormal_folders[negative_idx]
                csv_data.append([triplet_index, anchor_name, positive_name, negative_name])

                triplet_index += 1

    # Abnormal triplets
    for anchor_idx, anchor in enumerate(tqdm(abnormal_videos, desc="Creating abnormal triplets")):
        positive_indices = random.sample([i for i in range(len(abnormal_videos)) if i != anchor_idx], min(num_positive, len(abnormal_videos)-1))
        negative_indices = random.sample(range(len(normal_videos)), min(num_negative, len(normal_videos)))

        for positive_idx in positive_indices:
            for negative_idx in negative_indices:
                triplet_group = h5f.create_group(f'triplet_{triplet_index}')
                triplet_group.create_dataset('anchor', data=anchor)
                triplet_group.create_dataset('positive', data=abnormal_videos[positive_idx])
                triplet_group.create_dataset('negative', data=normal_videos[negative_idx])

                anchor_name = abnormal_folders[anchor_idx]
                positive_name = abnormal_folders[positive_idx]
                negative_name = normal_folders[negative_idx]
                csv_data.append([triplet_index, anchor_name, positive_name, negative_name])

                triplet_index += 1

    h5f.close()

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Triplet Index', 'Anchor', 'Positive', 'Negative'])
        csvwriter.writerows(csv_data)

    print(f"Saved triplet data to {output_path} and triplet info to {csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Process video frames to generate HDF5 files with uniform frame sampling.")
    parser.add_argument('--list-dir', type=str, required=True,
                        help='Path to the directory containing train.list and test.list.')
    parser.add_argument('--dataset', choices=["cuhk", "shanghaitech"], required=True,
                        help="Type of the dataset")
    parser.add_argument('--npy-dir', type=str, required=True,
                        help='Path to the directory containing video frame folders (training/frames and testing/frames).')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to the output directory where HDF5 files will be saved.')
    parser.add_argument('--num-pos', type=int, default=3, help='Number of positive samples per anchor(cuhk pos: 2~3 / shanghaitech pos: 5~7)')
    parser.add_argument('--num-neg', type=int, default=5, help='Number of negative samples per anchor(cuhk neg: 3~5 / shanghaitech neg: 10~15')
    args = parser.parse_args()

    train_list_file = os.path.join(args.list_dir, "train.list")
    test_list_file = os.path.join(args.list_dir, "test.list")

    with open(train_list_file, 'r') as file:
        train_video_list = file.read().strip().split('\n')

    with open(test_list_file, 'r') as file:
        test_video_list = file.read().strip().split('\n')

    train_video_folders = []
    test_video_folders = []

    for video_name in train_video_list:
        if args.dataset == "cuhk":
            train_video_folder = os.path.join(args.npy_dir, video_name)
        else:
            train_video_folder = os.path.join(args.npy_dir, "train", video_name)
        if os.path.isdir(train_video_folder):
            train_video_folders.append(train_video_folder)

    for video_name in test_video_list:
        if args.dataset == "cuhk":
            test_video_folder = os.path.join(args.npy_dir, video_name)
        else:
            test_video_folder = os.path.join(args.npy_dir, "test", video_name)

        if os.path.isdir(test_video_folder):
            test_video_folders.append(test_video_folder)

    # Split into normal and abnormal folders
    train_video_folders_normal = [folder for folder in train_video_folders if 'train' in folder]
    train_video_folders_abnormal = [folder for folder in train_video_folders if 'test' in folder]
    test_video_folders_normal = [folder for folder in test_video_folders if 'train' in folder]
    test_video_folders_abnormal = [folder for folder in test_video_folders if 'test' in folder]

    frame_counts = [16, 32, 64, 128]
    for count in frame_counts:
        print(f"Starting to generate HDF5 files for {count} frames (test: {len(test_video_folders_normal) + len(train_video_folders)} / train: {len(train_video_folders_normal) + len(train_video_folders_abnormal)})")

        train_output_path = os.path.join(args.output_dir, f'train_heatmap_f{count}.h5')
        save_h5_file(train_output_path, train_video_folders, count)

        test_output_path = os.path.join(args.output_dir, f'test_heatmap_f{count}.h5')
        save_h5_file(test_output_path, test_video_folders, count)

        train_triplet_h5_path = os.path.join(args.output_dir, f'train_triplet_heatmap_f{count}.h5')
        train_triplet_csv_path = os.path.join(args.output_dir, f'train_triplet_heatmap_f{count}.csv')
        create_triplet_h5(train_triplet_h5_path, train_video_folders_normal, train_video_folders_abnormal, count,
                          train_triplet_csv_path, num_positive=args.num_pos, num_negative=args.num_neg)

        test_triplet_h5_path = os.path.join(args.output_dir, f'test_triplet_heatmap_f{count}.h5')
        test_triplet_csv_path = os.path.join(args.output_dir, f'test_triplet_heatmap_f{count}.csv')
        create_triplet_h5(test_triplet_h5_path, test_video_folders_normal, test_video_folders_abnormal, count,
                          test_triplet_csv_path, num_positive=args.num_pos, num_negative=args.num_neg)


if __name__ == "__main__":
    main()
