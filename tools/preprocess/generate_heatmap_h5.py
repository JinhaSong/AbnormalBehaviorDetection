import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm


def load_npy_files(folder_path, num_frames):
    npy_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])
    total_files = len(npy_files)
    interval = max(total_files // num_frames, 1)

    selected_files = []
    for i in range(0, total_files, interval):
        selected_files.append(npy_files[i])
        if len(selected_files) == num_frames:
            break

    frames = [np.load(f) for f in selected_files]
    return np.array(frames)


def save_h5_file(output_path, dataset_dir, num_frames):
    h5f = h5py.File(output_path, 'w')

    video_folders = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if
                     os.path.isdir(os.path.join(dataset_dir, f))]

    for video_folder in tqdm(video_folders, desc=f"Processing {num_frames} frames"):
        frames = load_npy_files(video_folder, num_frames)

        group_name = os.path.basename(video_folder)
        h5f.create_dataset(group_name, data=frames)

    h5f.close()
    print(f"Saved {num_frames} frames per video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process video frames to generate HDF5 files with uniform frame sampling.")
    parser.add_argument('--dataset', choices=["cuhk", "shanghaitech", "ubnormal", "ucf"], required=True, help="Type of the dataset")
    parser.add_argument('--dataset-type', choices=["train", "test"], help='Type of the dataset(train or test)')
    parser.add_argument('--dataset-dir', type=str, help='Path to the dataset directory containing video frame folders.')
    parser.add_argument('--output-dir', type=str, help='Path to the output directory where HDF5 files will be saved.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    frame_counts = [16, 32, 64, 128]
    for count in frame_counts:
        output_path = os.path.join(args.output_dir, f'{args.dataset}_{args.dataset_type}_heatmap_f{count}.h5')
        save_h5_file(output_path, args.dataset_dir, count)


if __name__ == "__main__":
    main()
