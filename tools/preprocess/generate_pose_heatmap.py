import os
import sys
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.heatmap import GeneratePoseTarget


def save_heatmap_as_jpg(image, heatmap, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap, cmap='jet', alpha=1)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_heatmap(directory):
    pose_target_generator = GeneratePoseTarget(sigma=2.0)

    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
        [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
    ]

    json_files = [os.path.join(root, file)
                  for root, _, files in os.walk(directory)
                  for file in files if file.endswith('.json')]

    for json_path in tqdm(json_files, desc="Processing JSON files"):
        file = os.path.basename(json_path)
        image_path = os.path.join(os.path.dirname(json_path), file.replace('.json', '.jpg'))

        image = cv2.imread(image_path)
        if image is None:
            print(f"Image file not found for {file}, skipping...")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        pose_results = json_data['result']
        keypoints = []
        tracking_ids = []
        for person in pose_results:
            kp = [(float(p['x']), float(p['y'])) for p in person['pose'].values()]
            keypoints.append(kp)
            tracking_ids.append(person.get('tracking_id', 0))

        combined_heatmap = np.zeros((2, img_h, img_w), dtype=np.float32)
        for kp, tid in zip(keypoints, tracking_ids):
            kp = np.array(kp)
            for start, end in skeleton:
                start_point = kp[start]
                end_point = kp[end]
                heatmap = pose_target_generator.generate_a_limb_heatmap(img_h, img_w, start_point, end_point, 2.0, 1.0, 1.0)
                combined_heatmap[0] = np.maximum(combined_heatmap[0], heatmap)
                combined_heatmap[1][heatmap > 0] = tid

        npy_output_path = os.path.join(os.path.dirname(json_path), file.replace('.json', '.npy'))
        np.save(npy_output_path, combined_heatmap)

        jpg_output_path = os.path.join(os.path.dirname(json_path), file.replace('.json', '_heatmap.jpg'))
        save_heatmap_as_jpg(image_rgb, combined_heatmap[0], jpg_output_path)


def main():
    parser = argparse.ArgumentParser(description="Process video frames to generate pose heatmaps.")
    parser.add_argument('--dataset-dir', type=str, help='Path to the dataset directory containing frames and JSON files.')
    args = parser.parse_args()

    generate_heatmap(args.dataset_dir)

if __name__ == "__main__":
    main()