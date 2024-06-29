import os
import csv
import h5py
import random
import argparse
import numpy as np
from tqdm import tqdm

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

    frames = [np.load(f) for f in selected_files]
    return np.array(frames), np.array(frame_numbers)

def save_h5_file(output_path, video_folders, num_frames, is_anomaly):
    h5f = h5py.File(output_path, 'w')

    for video_folder in tqdm(video_folders, desc=f"Processing {num_frames} frames"):
        if not os.path.exists(video_folder):
            print(f"Video folder {video_folder} does not exist. Skipping.")
            continue

        frames, frame_numbers = load_npy_files(video_folder, num_frames)

        group_name = os.path.basename(video_folder)
        group = h5f.create_group(group_name)
        group.create_dataset('video_data', data=frames)
        group.create_dataset('frame_numbers', data=frame_numbers)

        label = 1 if is_anomaly else 0
        group.create_dataset('label', data=label)

    h5f.close()
    print(f"Saved {num_frames} frames per video to {output_path}")

def create_triplet_h5(output_path, normal_folders, abnormal_folders, num_frames, num_pos, num_neg, csv_path):
    video_list = []
    triplet_info = []
    normal_indices = []
    abnormal_indices = []

    with h5py.File(output_path, 'w') as h5f:
        videos_grp = h5f.create_group('videos')

        # Normal videos
        for i, folder in enumerate(tqdm(normal_folders, desc="Adding normal videos")):
            video, _ = load_npy_files(folder, num_frames)
            videos_grp.create_dataset(f'{i}', data=video)
            normal_indices.append(i)
            video_list.append((i, os.path.basename(folder)))

        # Abnormal videos
        abnormal_start_idx = len(normal_folders)
        for i, folder in enumerate(tqdm(abnormal_folders, desc="Adding abnormal videos")):
            video, _ = load_npy_files(folder, num_frames)
            videos_grp.create_dataset(f'{i + abnormal_start_idx}', data=video)
            abnormal_indices.append(i + abnormal_start_idx)
            video_list.append((i + abnormal_start_idx, os.path.basename(folder)))

        # Save indices
        h5f.create_dataset('normal_indices', data=np.array(normal_indices, dtype=np.int32))
        h5f.create_dataset('abnormal_indices', data=np.array(abnormal_indices, dtype=np.int32))

        # Normal triplets
        for anchor_idx in tqdm(normal_indices, desc="Creating normal triplets"):
            positive_indices = random.sample([i for i in normal_indices if i != anchor_idx], min(num_pos, len(normal_indices)-1))
            negative_indices = random.sample(abnormal_indices, min(num_neg, len(abnormal_indices)))
            triplet_info.extend([(anchor_idx, pos_idx, neg_idx) for pos_idx in positive_indices for neg_idx in negative_indices])

        # Abnormal triplets
        for anchor_idx in tqdm(abnormal_indices, desc="Creating abnormal triplets"):
            positive_indices = random.sample([i for i in abnormal_indices if i != anchor_idx], min(num_pos, len(abnormal_indices)-1))
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
    parser = argparse.ArgumentParser(
        description="Process video frames to generate HDF5 files with uniform frame sampling.")
    parser.add_argument('--npy-dir', type=str, required=True,
                        help='Path to the directory containing video frame npy folders (training/frames and testing/frames).')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to the output directory where HDF5 files will be saved.')
    parser.add_argument('--num-pos', type=int, default=3, help='Number of positive samples per anchor(ubnormal: 5)')
    parser.add_argument('--num-neg', type=int, default=5, help='Number of negative samples per anchor(ubnormal: 10)')
    args = parser.parse_args()

    train_frames_dir = os.path.join(args.npy_dir, 'train')
    test_frames_dir = os.path.join(args.npy_dir, 'test')

    train_video_folders_normal = [os.path.join(train_frames_dir, d) for d in os.listdir(train_frames_dir) if os.path.isdir(os.path.join(train_frames_dir, d)) and 'normal' in d]
    train_video_folders_abnormal = [os.path.join(train_frames_dir, d) for d in os.listdir(train_frames_dir) if os.path.isdir(os.path.join(train_frames_dir, d)) and 'abnormal' in d]
    test_video_folders_normal = [os.path.join(test_frames_dir, d) for d in os.listdir(test_frames_dir) if os.path.isdir(os.path.join(test_frames_dir, d)) and 'normal' in d]
    test_video_folders_abnormal = [os.path.join(test_frames_dir, d) for d in os.listdir(test_frames_dir) if os.path.isdir(os.path.join(test_frames_dir, d)) and 'abnormal' in d]

    frame_counts = [64, 128]
    for count in frame_counts:
        print(f"Starting to generate HDF5 files for {count} frames")

        # Create triplet index data for contrastive learning
        train_triplet_h5_path = os.path.join(args.output_dir, f'ubnormal_train_triplet_index_f{count}.h5')
        create_triplet_h5(train_triplet_h5_path, train_video_folders_normal, train_video_folders_abnormal, count, args.num_pos, args.num_neg, args.output_dir)

        test_triplet_h5_path = os.path.join(args.output_dir, f'ubnormal_test_triplet_index_f{count}.h5')
        create_triplet_h5(test_triplet_h5_path, test_video_folders_normal, test_video_folders_abnormal, count, args.num_pos, args.num_neg, args.output_dir)

if __name__ == "__main__":
    main()
