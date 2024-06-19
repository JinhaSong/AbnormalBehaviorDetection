import os
import sys
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        return joint_and_limb_heatmap  # Skip if bbox width or height is 0

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

    json_files = sorted([os.path.join(root, file)
                         for root, _, files in os.walk(input_directory)
                         for file in files if file.endswith('.json')])

    for json_path in tqdm(json_files, desc="Processing JSON files"):
        file = os.path.basename(json_path)
        image_path = os.path.join(os.path.dirname(json_path), file.replace('.json', '.jpg'))

        image = cv2.imread(image_path)
        if image is None:
            print(f"Image file not found for {file}, skipping...")
            continue

        img_h, img_w = image.shape[:2]

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

        # Resize each heatmap to 224x224 and create a 32x224x224 tensor
        resized_heatmaps = np.zeros((32, 224, 224), dtype=np.float32)
        for i in range(min(max_human_objects, 32)):
            resized_heatmaps[i] = cv2.resize(combined_heatmaps[i], (224, 224))

        # Sum all heatmaps into a single image for visualization
        full_heatmap = np.sum(combined_heatmaps, axis=0)

        # Generate output path maintaining the directory structure
        relative_path = os.path.relpath(json_path, input_directory)
        npy_output_path = os.path.join(output_directory, os.path.splitext(relative_path)[0] + '.npy')
        os.makedirs(os.path.dirname(npy_output_path), exist_ok=True)
        np.save(npy_output_path, resized_heatmaps)

        # Visualize the concatenated heatmap
        jpg_output_path = os.path.join(output_directory, os.path.splitext(relative_path)[0] + '_heatmap.jpg')
        os.makedirs(os.path.dirname(jpg_output_path), exist_ok=True)
        save_heatmap_as_jpg(full_heatmap, jpg_output_path)


def main():
    parser = argparse.ArgumentParser(description="Process video frames to generate pose heatmaps.")
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to the input dataset directory containing frames and JSON files.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to the output directory where NPZ files will be saved.')
    parser.add_argument('--max-human-objects', type=int, required=True,
                        help='Maximum number of human objects to consider in each frame.')
    args = parser.parse_args()

    generate_heatmap(args.input_dir, args.output_dir, args.max_human_objects)


if __name__ == "__main__":
    main()
