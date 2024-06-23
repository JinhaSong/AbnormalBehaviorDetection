import os
import sys
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.heatmap import GeneratePoseTarget

def save_heatmap_as_jpg(heatmap, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap, cmap='jet', alpha=1)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_individual_heatmap(image, keypoints, bbox, skeleton, pose_target_generator, margin=10):
    img_h, img_w = image.shape[:2]
    joint_and_limb_heatmap = np.zeros((img_h, img_w), dtype=np.float32)

    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    if w == 0 or h == 0:
        return joint_and_limb_heatmap

    x_min = max(0, x - margin)
    y_min = max(0, y - margin)
    x_max = min(img_w, x + w + margin)
    y_max = min(img_h, y + h + margin)
    bbox_w, bbox_h = x_max - x_min, y_max - y_min

    limb_heatmap = np.zeros((bbox_h, bbox_w), dtype=np.float32)
    for start, end in skeleton:
        start_point = np.array(keypoints[start]) - [x_min, y_min]
        end_point = np.array(keypoints[end]) - [x_min, y_min]
        heatmap = pose_target_generator.generate_a_limb_heatmap(bbox_h, bbox_w, start_point, end_point, 2.0, 1.0, 1.0)
        limb_heatmap = np.maximum(limb_heatmap, heatmap)

    joint_and_limb_heatmap[y_min:y_max, x_min:x_max] = np.maximum(joint_and_limb_heatmap[y_min:y_max, x_min:x_max],
                                                                  limb_heatmap)

    return joint_and_limb_heatmap

def generate_heatmap(input_directory, output_directory, max_human_objects):
    pose_target_generator = GeneratePoseTarget(sigma=2.0)

    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
        [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
    ]

    video_lengths = []
    resolutions = []
    num_objects_list = []

    for video_folder in sorted(os.listdir(input_directory)):
        video_folder_path = os.path.join(input_directory, video_folder)
        if not os.path.isdir(video_folder_path):
            continue

        json_files = sorted([os.path.join(video_folder_path, file)
                             for file in os.listdir(video_folder_path) if file.endswith('.json')])

        if not json_files:
            continue

        num_frames = len(json_files)
        video_lengths.append(num_frames)

        image_path = json_files[0].replace('.json', '.jpg')
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image file not found for {image_path}, skipping...")
            continue

        img_h, img_w = image.shape[:2]
        resolutions.append((img_h, img_w))

        for json_path in tqdm(json_files, desc=f"Processing {video_folder}"):
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            pose_results = json_data['result']
            keypoints = []
            bboxes = []
            for person in pose_results:
                kp = [(float(p['x']), float(p['y'])) for p in person['pose'].values()]
                bbox = person['position']['x'], person['position']['y'], person['position']['w'], person['position']['h']
                keypoints.append(kp)
                bboxes.append(bbox)

            combined_heatmaps = np.zeros((max_human_objects, img_h, img_w), dtype=np.float32)

            for i, (kp, bbox) in enumerate(zip(keypoints, bboxes)):
                if i >= max_human_objects:
                    break
                kp = np.array(kp)
                individual_heatmap = generate_individual_heatmap(image, kp, bbox, skeleton, pose_target_generator)
                combined_heatmaps[i] = individual_heatmap

            resized_heatmaps = np.zeros((max_human_objects, 224, 224), dtype=np.float32)
            for i in range(min(max_human_objects, max_human_objects)):
                resized_heatmaps[i] = cv2.resize(combined_heatmaps[i], (224, 224))

            for i in range(min(max_human_objects, max_human_objects), max_human_objects):
                resized_heatmaps[i] = -np.ones((224, 224), dtype=np.float32)

            full_heatmap = np.sum(combined_heatmaps, axis=0)

            relative_path = os.path.relpath(json_path, input_directory)
            npy_output_path = os.path.join(output_directory, os.path.splitext(relative_path)[0] + '.npy')
            os.makedirs(os.path.dirname(npy_output_path), exist_ok=True)
            np.save(npy_output_path, resized_heatmaps)

            jpg_output_path = os.path.join(output_directory, os.path.splitext(relative_path)[0] + '_heatmap.jpg')
            os.makedirs(os.path.dirname(jpg_output_path), exist_ok=True)
            save_heatmap_as_jpg(full_heatmap, jpg_output_path)

        num_objects_list.append(len(pose_results))

    return video_lengths, resolutions, num_objects_list

def main():
    parser = argparse.ArgumentParser(description="Process video frames to generate pose heatmaps.")
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to the input dataset directory containing frames and JSON files.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to the output directory where NPZ files will be saved.')
    parser.add_argument('--max-human-objects', type=int, required=True,
                        help='Maximum number of human objects to consider in each frame.')
    args = parser.parse_args()

    all_video_lengths = []
    all_resolutions = []
    all_num_objects = []

    for dataset_type in ["train", "test"]:
        video_lengths, resolutions, num_objects_list = generate_heatmap(
            os.path.join(args.input_dir, dataset_type),
            os.path.join(args.output_dir, dataset_type),
            args.max_human_objects
        )
        all_video_lengths.extend(video_lengths)
        all_resolutions.extend(resolutions)
        all_num_objects.extend(num_objects_list)

        with open(os.path.join(args.output_dir, f'{dataset_type}.csv'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Video', 'Length (frames)', 'Length (seconds)', 'Resolution (H, W)', 'Number of objects'])
            for i, (length, resolution, num_objects) in enumerate(zip(video_lengths, resolutions, num_objects_list)):
                length_seconds = length / 30.0  # Assuming 30 FPS
                csvwriter.writerow([f'video_{i}', length, length_seconds, resolution, num_objects])

    total_frames = sum(all_video_lengths)
    avg_video_length = total_frames / len(all_video_lengths) if all_video_lengths else 0
    avg_video_length_sec = avg_video_length / 30  # Assuming 30 FPS

    print(f'Total frames: {total_frames}')
    print(f'Average video length (frames): {avg_video_length}')
    print(f'Average video length (seconds): {avg_video_length_sec}')

if __name__ == "__main__":
    main()
