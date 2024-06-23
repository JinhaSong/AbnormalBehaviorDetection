import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm
import csv


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

        # Set label based on the folder name
        label = 1 if is_anomaly else 0
        group.create_dataset('label', data=label)

    h5f.close()
    print(f"Saved {num_frames} frames per video to {output_path}")


def create_triplet_h5(output_path, normal_folders, abnormal_folders, num_frames, csv_path):
    h5f = h5py.File(output_path, 'w')
    csv_data = []

    normal_videos = [load_npy_files(folder, num_frames)[0] for folder in normal_folders]
    abnormal_videos = [load_npy_files(folder, num_frames)[0] for folder in abnormal_folders]

    triplet_index = 0

    # Normal triplets
    for anchor in normal_videos:
        for positive in normal_videos:
            if not np.array_equal(anchor, positive):
                for negative in abnormal_videos:
                    triplet_group = h5f.create_group(f'triplet_{triplet_index}')
                    triplet_group.create_dataset('anchor', data=anchor)
                    triplet_group.create_dataset('positive', data=positive)
                    triplet_group.create_dataset('negative', data=negative)

                    # Save triplet info to CSV data
                    anchor_name = normal_folders[normal_videos.index(anchor)]
                    positive_name = normal_folders[normal_videos.index(positive)]
                    negative_name = abnormal_folders[abnormal_videos.index(negative)]
                    csv_data.append([triplet_index, anchor_name, positive_name, negative_name])

                    triplet_index += 1

    # Abnormal triplets
    for anchor in abnormal_videos:
        for positive in abnormal_videos:
            if not np.array_equal(anchor, positive):
                for negative in normal_videos:
                    triplet_group = h5f.create_group(f'triplet_{triplet_index}')
                    triplet_group.create_dataset('anchor', data=anchor)
                    triplet_group.create_dataset('positive', data=positive)
                    triplet_group.create_dataset('negative', data=negative)

                    # Save triplet info to CSV data
                    anchor_name = abnormal_folders[abnormal_videos.index(anchor)]
                    positive_name = abnormal_folders[abnormal_videos.index(positive)]
                    negative_name = normal_folders[normal_videos.index(negative)]
                    csv_data.append([triplet_index, anchor_name, positive_name, negative_name])

                    triplet_index += 1

    h5f.close()

    # Save triplet info to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Triplet Index', 'Anchor', 'Positive', 'Negative'])
        csvwriter.writerows(csv_data)

    print(f"Saved triplet data to {output_path} and triplet info to {csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Process video frames to generate HDF5 files with uniform frame sampling.")
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to the directory containing video frame folders (training/frames and testing/frames).')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to the output directory where HDF5 files will be saved.')
    parser.add_argument('--frame-counts', type=int, nargs='+', default=[64, 128],
                        help='List of frame counts to generate heatmaps for.')
    args = parser.parse_args()

    train_frames_dir = os.path.join(args.dataset_dir, 'train')
    test_frames_dir = os.path.join(args.dataset_dir, 'test')

    train_video_folders_normal = [os.path.join(train_frames_dir, d) for d in os.listdir(train_frames_dir) if
                                  os.path.isdir(os.path.join(train_frames_dir, d)) and 'normal' in d]
    train_video_folders_abnormal = [os.path.join(train_frames_dir, d) for d in os.listdir(train_frames_dir) if
                                    os.path.isdir(os.path.join(train_frames_dir, d)) and 'abnormal' in d]
    test_video_folders_normal = [os.path.join(test_frames_dir, d) for d in os.listdir(test_frames_dir) if
                                 os.path.isdir(os.path.join(test_frames_dir, d)) and 'normal' in d]
    test_video_folders_abnormal = [os.path.join(test_frames_dir, d) for d in os.listdir(test_frames_dir) if
                                   os.path.isdir(os.path.join(test_frames_dir, d)) and 'abnormal' in d]

    for count in args.frame_counts:
        print(f"Starting to generate HDF5 files for {count} frames")

        # Train data
        train_output_path_normal = os.path.join(args.output_dir, f'ubnormal_train_heatmap_normal_f{count}.h5')
        train_output_path_abnormal = os.path.join(args.output_dir, f'ubnormal_train_heatmap_abnormal_f{count}.h5')
        save_h5_file(train_output_path_normal, train_video_folders_normal, count, is_anomaly=False)
        save_h5_file(train_output_path_abnormal, train_video_folders_abnormal, count, is_anomaly=True)

        # Test data
        test_output_path_normal = os.path.join(args.output_dir, f'ubnormal_test_heatmap_normal_f{count}.h5')
        test_output_path_abnormal = os.path.join(args.output_dir, f'ubnormal_test_heatmap_abnormal_f{count}.h5')
        save_h5_file(test_output_path_normal, test_video_folders_normal, count, is_anomaly=False)
        save_h5_file(test_output_path_abnormal, test_video_folders_abnormal, count, is_anomaly=True)

        # Create triplet data for contrastive learning
        train_triplet_h5_path = os.path.join(args.output_dir, f'ubnormal_train_triplet_heatmap_f{count}.h5')
        train_triplet_csv_path = os.path.join(args.output_dir, f'ubnormal_train_triplet_heatmap_f{count}.csv')
        create_triplet_h5(train_triplet_h5_path, train_video_folders_normal, train_video_folders_abnormal, count,
                          train_triplet_csv_path)

        test_triplet_h5_path = os.path.join(args.output_dir, f'ubnormal_test_triplet_heatmap_f{count}.h5')
        test_triplet_csv_path = os.path.join(args.output_dir, f'ubnormal_test_triplet_heatmap_f{count}.csv')
        create_triplet_h5(test_triplet_h5_path, test_video_folders_normal, test_video_folders_abnormal, count,
                          test_triplet_csv_path)


if __name__ == "__main__":
    main()
