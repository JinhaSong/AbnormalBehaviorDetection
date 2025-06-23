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


def normalize_tracking_id(json_data):
    """tracking_id의 최솟값을 1로 맞추도록 정규화"""
    tracking_ids = [person['tracking_id'] for person in json_data['result']]
    min_tracking_id = min(tracking_ids)

    # 모든 tracking_id를 1부터 시작하게 조정
    for person in json_data['result']:
        person['tracking_id'] = person['tracking_id'] - min_tracking_id + 1


def process_json_and_normalize(json_folder):
    """json 폴더를 읽어 tracking_id를 정규화하고 수정된 json을 다시 저장"""
    json_files = sorted([os.path.join(json_folder, file) for file in os.listdir(json_folder) if file.endswith('.json')])

    if not json_files:
        return

    for json_path in json_files:
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # tracking_id를 1부터 시작하도록 수정
        normalize_tracking_id(json_data)

        # 수정된 json을 다시 저장
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)


def calculate_max_human_objects(input_directory):
    """모든 JSON 파일을 읽어 최대 객체 수를 계산하고, 이를 2의 제곱수로 매핑"""
    max_objects = 0

    for root, _, files in os.walk(input_directory):
        json_files = [os.path.join(root, f) for f in files if f.endswith('.json')]

        for json_file in json_files:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                num_objects = len(json_data['result'])
                if num_objects > max_objects:
                    max_objects = num_objects

    # 가장 가까운 2의 제곱수로 매핑
    return 1 << (max_objects - 1).bit_length()


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

        # JSON 파일을 정제하여 tracking_id를 정규화
        process_json_and_normalize(video_folder_path)

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
                bbox = person['position']['x'], person['position']['y'], person['position']['w'], person['position'][
                    'h']
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


def user_confirmation(prompt):
    """사용자에게 y/n 입력을 받고, y일 경우 True 반환"""
    while True:
        answer = input(f"{prompt} (y/n): ").strip().lower()
        if answer == 'y':
            return True
        elif answer == 'n':
            return False
        else:
            print("Invalid input, please enter 'y' or 'n'.")


def select_datasets_and_classes(input_dir):
    """데이터셋 및 클래스 선택"""
    selected_datasets = []

    for dataset_type in sorted(os.listdir(input_dir)):
        dataset_dir = os.path.join(input_dir, dataset_type)
        if not os.path.isdir(dataset_dir):
            continue

        # 데이터셋을 실행할지 여부 확인
        if user_confirmation(f"Do you want to proceed {dataset_type}?"):
            selected_classes = []
            for cls in sorted(os.listdir(dataset_dir)):
                class_dir = os.path.join(dataset_dir, cls)
                if not os.path.isdir(class_dir):
                    continue
                # 클래스별로 실행할지 여부 확인
                if user_confirmation(f"    - Do you want to proceed {cls} in {dataset_type}?"):
                    selected_classes.append(cls)

            if selected_classes:
                selected_datasets.append((dataset_type, selected_classes))

    return selected_datasets


def main():
    parser = argparse.ArgumentParser(description="Process video frames to generate pose heatmaps.")
    parser.add_argument('--clip-dir', type=str, required=True,
                        help='Path to the clip directory containing frames and JSON files.')
    args = parser.parse_args()

    # 사용자에게 각 디렉토리에 대해 실행할지 여부를 확인
    selected_datasets = select_datasets_and_classes(os.path.join(args.clip_dir, "videos"))

    all_video_lengths = []
    all_resolutions = []
    all_num_objects = []

    for dataset_type, selected_classes in selected_datasets:
        for cls in selected_classes:
            input_dir = os.path.join(args.clip_dir, "videos", dataset_type, cls)
            output_dir = os.path.join(args.clip_dir, "npy", dataset_type, cls)

            # 최대 객체 수 계산
            max_human_objects = calculate_max_human_objects(input_dir)

            video_lengths, resolutions, num_objects_list = generate_heatmap(
                input_dir,
                output_dir,
                max_human_objects
            )
            all_video_lengths.extend(video_lengths)
            all_resolutions.extend(resolutions)
            all_num_objects.extend(num_objects_list)

            # CSV 파일을 npy 디렉토리 안에 저장
            with open(os.path.join(output_dir, f'{dataset_type}_{cls}.csv'), 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(
                    ['Video', 'Length (frames)', 'Length (seconds)', 'Resolution (H, W)', 'Number of objects'])
                for i, (length, resolution, num_objects) in enumerate(
                        zip(video_lengths, resolutions, num_objects_list)):
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
