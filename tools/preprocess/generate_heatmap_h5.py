import os
import csv
import h5py
import json
import argparse
import numpy as np
from tqdm import tqdm


def resize_frame(frame, target_size=(2, 224, 224)):
    return np.resize(frame, target_size)


def load_npy_files(folder_path, frame_numbers):
    frames = []
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    for frame_number in frame_numbers:
        file_path = os.path.join(folder_path, npy_files[frame_number])
        frame = np.load(file_path)
        frames.append(resize_frame(frame))
    return np.array(frames)


def calculate_anomaly_stats(annotation, min_length=64):
    all_lengths = []
    for video in annotation:
        intervals = annotation[video]['anomalies']
        for anomaly_dict in intervals:
            for anomaly_type in anomaly_dict:
                for interval in anomaly_dict[anomaly_type]:
                    start, end = interval
                    if end - start >= min_length:
                        all_lengths.append(end - start)

    if not all_lengths:
        print("No anomaly intervals meet the minimum length requirement.")
        return 0, 0

    max_length = max(all_lengths)
    min_length = min(all_lengths)
    avg_length = int(sum(all_lengths) / len(all_lengths))
    print(f"Max Anomaly Length: {max_length}")
    print(f"Min Anomaly Length: {min_length}")
    print(f"Avg Anomaly Length: {avg_length}")
    return max_length, avg_length


def parse_annotations(dataset, dataset_dir):
    if dataset in ["cuhk", "shanghaitech"]:
        annotation_path = os.path.join(dataset_dir, 'annotations.json')
        print(annotation_path)
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = {}
    return annotations


def generate_segments(video_length, segment_length):
    step = segment_length // 2
    segments = []
    for start in range(0, video_length - segment_length + 1, step):
        end = start + segment_length
        segments.append((start, end))
    return segments


def interpolate_frames(frames, target_length):
    current_length = len(frames)
    if current_length >= target_length:
        return frames

    indices = np.linspace(0, current_length - 1, target_length).astype(int)
    interpolated_frames = frames[indices]
    return interpolated_frames


def create_anomaly_segments(annotations, annotation_key, min_length):
    anomaly_intervals = []
    intervals = annotations[annotation_key]['anomalies']
    for anomaly_dict in intervals:
        for anomaly_type in anomaly_dict:
            for interval in anomaly_dict[anomaly_type]:
                start, end = interval
                if end - start >= min_length:
                    anomaly_intervals.append((start, end))
    return anomaly_intervals


def create_normal_segments(anomaly_intervals, video_length, min_length):
    normal_intervals = []
    prev_end = 0
    for start, end in sorted(anomaly_intervals):
        if prev_end < start:
            normal_intervals.append((prev_end, start))
        prev_end = end
    if prev_end < video_length:
        normal_intervals.append((prev_end, video_length))

    normal_segments = []
    for start, end in normal_intervals:
        if end - start >= min_length:
            normal_segments.append((start, end))
        else:
            continue
    return normal_segments


def uniform_sampling(folder_path, segment, num_frames):
    start, end = segment
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    frame_indices = np.linspace(start, end - 1, num_frames).astype(int)
    frames = load_npy_files(folder_path, frame_indices)
    frame_numbers = frame_indices + start
    return frames, frame_numbers


def save_test_h5_file(output_path, dataset_dir, num_frames, annotations, min_length, csv_path, json_path):
    h5f = h5py.File(output_path, 'w')
    metadata = []

    video_folders = sorted(
        [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])

    for video_folder in tqdm(video_folders, desc=f"Processing {num_frames} frames"):
        group_name = os.path.basename(video_folder)
        annotation_key = group_name.split('_')[-1]
        if annotation_key not in annotations:
            print(f"Warning: {annotation_key} not found in annotations. Skipping.")
            continue

        video_length = len([f for f in os.listdir(video_folder) if f.endswith('.npy')])

        anomaly_intervals = create_anomaly_segments(annotations, annotation_key, min_length)
        normal_segments = create_normal_segments(anomaly_intervals, video_length, min_length)

        for start, end in anomaly_intervals + normal_segments:
            if end - start < min_length:
                continue

            frames, frame_numbers = uniform_sampling(video_folder, (start, end), num_frames)

            if num_frames == 128 and len(frame_numbers) < 128:
                frames = interpolate_frames(frames, 128)
                frame_numbers = np.linspace(start, end, 128).astype(int)

            label = 1 if (start, end) in anomaly_intervals else 0  # 1 for anomaly, 0 for normal

            segment_metadata = {
                "video_name": f"{group_name}_{start}_{end}",
                "video_length": end - start,
                "label": "anomaly" if label == 1 else "normal",
                "original_video": {
                    "video_name": group_name,
                    "original_video_segment": {
                        "start": start,
                        "end": end
                    }
                }
            }
            metadata.append(segment_metadata)

            group = h5f.create_group(f"{group_name}_{start}_{end}")
            group.create_dataset('video_data', data=frames)
            group.create_dataset('frame_numbers', data=frame_numbers)
            group.create_dataset('label', data=label)
            group.create_dataset('annotation', data=json.dumps(segment_metadata))  # Add annotation dataset

    h5f.close()
    print(f"Saved {num_frames} frames per video to {output_path}")

    # Save metadata to JSON and CSV
    if metadata:
        with open(json_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
            print(f"Saved metadata to {json_path}")

        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["video_name", "video_length", "label", "original_video_name",
                                                          "original_video_segment_start", "original_video_segment_end"])
            writer.writeheader()
            for data in metadata:
                writer.writerow({
                    "video_name": data["video_name"],
                    "video_length": data["video_length"],
                    "label": data["label"],
                    "original_video_name": data["original_video"]["video_name"],
                    "original_video_segment_start": data["original_video"]["original_video_segment"]["start"],
                    "original_video_segment_end": data["original_video"]["original_video_segment"]["end"]
                })
            print(f"Saved metadata to {csv_path}")
    else:
        print("No metadata to save.")


def save_train_h5_file(output_path, dataset_dir, num_frames, avg_length, min_length, csv_path, json_path):
    h5f = h5py.File(output_path, 'w')
    metadata = []

    video_folders = sorted(
        [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])

    for video_folder in tqdm(video_folders, desc=f"Processing {num_frames} frames for training"):
        group_name = os.path.basename(video_folder)
        video_length = len([f for f in os.listdir(video_folder) if f.endswith('.npy')])

        segments = generate_segments(video_length, avg_length)
        for start, end in segments:
            if end - start < min_length:
                continue

            frames, frame_numbers = uniform_sampling(video_folder, (start, end), num_frames)
            if num_frames == 128 and len(frame_numbers) < 128:
                frames = interpolate_frames(frames, 128)
                frame_numbers = np.linspace(start, end, 128).astype(int)

            segment_metadata = {
                "video_name": f"{group_name}_{start}_{end}",
                "video_length": end - start,
                "original_video": {
                    "video_name": group_name,
                    "original_video_segment": {
                        "start": start,
                        "end": end
                    }
                }
            }
            metadata.append(segment_metadata)

            group = h5f.create_group(f"{group_name}_{start}_{end}")
            group.create_dataset('video_data', data=frames)
            group.create_dataset('frame_numbers', data=frame_numbers)
            group.create_dataset('annotation', data=json.dumps(segment_metadata))  # Add annotation dataset

    h5f.close()
    print(f"Saved {num_frames} frames per video to {output_path}")

    # Save metadata to JSON and CSV
    if metadata:
        with open(json_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
            print(f"Saved metadata to {json_path}")

        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["video_name", "video_length", "original_video_name",
                                                          "original_video_segment_start", "original_video_segment_end"])
            writer.writeheader()
            for data in metadata:
                writer.writerow({
                    "video_name": data["video_name"],
                    "video_length": data["video_length"],
                    "original_video_name": data["original_video"]["video_name"],
                    "original_video_segment_start": data["original_video"]["original_video_segment"]["start"],
                    "original_video_segment_end": data["original_video"]["original_video_segment"]["end"]
                })
            print(f"Saved metadata to {csv_path}")
    else:
        print("No metadata to save.")


def process_videos(output_dir, dataset_dir, frame_counts, dataset, dataset_type):
    os.makedirs(output_dir, exist_ok=True)

    annotations = parse_annotations(dataset, dataset_dir)
    if dataset in ["cuhk", "shanghaitech"]:
        max_length, avg_length = calculate_anomaly_stats(annotations)

    min_length = 64  # Set minimum segment length to 64

    for count in frame_counts:
        output_path = os.path.join(output_dir, f'{dataset}_{dataset_type}_heatmap_f{count}.h5')
        json_path = os.path.join(output_dir, f'{dataset}_{dataset_type}_heatmap_f{count}.json')
        csv_path = os.path.join(output_dir, f'{dataset}_{dataset_type}_heatmap_f{count}.csv')

        if dataset_type == "test":
            save_test_h5_file(output_path, dataset_dir, count, annotations, min_length, csv_path, json_path)
        else:
            save_train_h5_file(output_path, dataset_dir, count, avg_length, min_length, csv_path, json_path)


def main():
    parser = argparse.ArgumentParser(
        description="Process video frames to generate HDF5 files with uniform frame sampling.")
    parser.add_argument('--dataset', choices=["cuhk", "shanghaitech", "ubnormal", "ucf"], required=True,
                        help="Type of the dataset")
    parser.add_argument('--dataset-type', choices=["train", "test"], help='Type of the dataset(train or test)')
    parser.add_argument('--dataset-dir', type=str, help='Path to the dataset directory containing video frame folders.')
    parser.add_argument('--output-dir', type=str, help='Path to the output directory where HDF5 files will be saved.')
    args = parser.parse_args()

    frame_counts = [16, 32, 64, 128]
    process_videos(args.output_dir, args.dataset_dir, frame_counts, args.dataset, args.dataset_type)


if __name__ == "__main__":
    main()
