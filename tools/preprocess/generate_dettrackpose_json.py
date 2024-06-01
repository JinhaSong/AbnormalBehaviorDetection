import os
import sys
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.cfg import load_cfg
from util.dataset import parse_dataset_dir, parse_video_list
from util.vis import Visualization
from lib.yolo.yolov7.det import YOLOv7
from lib.pose.alphapose.pose import AlphaPose
from lib.pose.simplehrnet.main import SimpleHRNetWrapper
from lib.tracker.bytetracker.BYTETracker import BYTETracker

vis = Visualization()

def find_and_convert_int_values(data):
    """
    Recursively find and convert integer values in a nested dictionary or list.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = find_and_convert_int_values(value)
    elif isinstance(data, list):
        data = [find_and_convert_int_values(item) for item in data]
    elif isinstance(data, (np.int64, np.int32, int)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    return data

def save_results(frame, frame_output_dir, frame_name, pose_object_result, video_name, frame_number):
    os.makedirs(frame_output_dir, exist_ok=True)
    frame_path = os.path.join(frame_output_dir, f"{frame_name}.jpg")
    cv2.imwrite(frame_path, frame)

    formatted_result = {
        "video_name": video_name,
        "frame_number": frame_number,
        "result": find_and_convert_int_values(pose_object_result)
    }

    frame_bbox = vis.draw_bboxes_person(frame, pose_object_result, 0.1)
    frame_pose = vis.draw_points_and_skeleton(frame_bbox, pose_object_result)
    result_path = os.path.join(frame_output_dir, f"{frame_name}_result.jpg")
    cv2.imwrite(result_path, frame_pose)

    json_path = os.path.join(frame_output_dir, f"{frame_name}.json")
    with open(json_path, "w") as json_file:
        json.dump(formatted_result, json_file, indent=4)


def process_frames(frames, output_dir, det_model, track_model, pose_model, video_name, start_frame_number):
    detection_results = det_model.inference_image_batch(frames)
    tracking_results = [track_model.update(result) for result in detection_results]

    for i, (frame, tracking_result) in enumerate(zip(frames, tracking_results)):
        frame_number = start_frame_number + i
        frame_name = f"{video_name}_{frame_number:05d}"
        pose_object_result = pose_model.inference_image(frame, tracking_result)

        save_results(frame, output_dir, frame_name, pose_object_result, video_name, frame_number)


def extract_and_process_frames(video_path, output_dir, frame_prefix, det_model, track_model, pose_model, batch_size):
    os.makedirs(output_dir, exist_ok=True)
    video_path_str = str(video_path)
    vidcap = cv2.VideoCapture(video_path_str)
    if not vidcap.isOpened():
        print(f"Error: Unable to open video file {video_path_str}")
        return

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc=f"Processing frames from {video_path.stem}") as pbar:
        frames = []
        count = 0
        success, frame = vidcap.read()
        while success:
            frames.append(frame)
            if len(frames) == batch_size:
                process_frames(frames, output_dir, det_model, track_model, pose_model, frame_prefix, count)
                frames = []

            success, frame = vidcap.read()
            count += 1
            pbar.update(1)

        # Process any remaining frames
        if frames:
            process_frames(frames, output_dir, det_model, track_model, pose_model, frame_prefix, count - len(frames))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--device", type=str, default="0", help="GPU device number")
    parser.add_argument("--det-cfg", type=str, default="cfgs/cfg_yolov7x.yml",
                        help="Path of object detection parameter")
    parser.add_argument("--pose-cfg", type=str, default="cfgs/cfg_hrnet.yml",
                        help="Path of pose estimation parameter")
    parser.add_argument("--track-cfg", type=str, default="cfgs/cfg_bytetrack.yml",
                        help="Path of object tracking parameter")
    parser.add_argument("--dataset-dir", required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset", choices=["cuhk", "shanghaitech", "ubnormal", "ucf"], required=True,
                        help="Type of the dataset")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing frames")

    opt = parser.parse_known_args()[0]

    device = opt.device
    det_cfg = load_cfg(opt.det_cfg)["infer"]
    pose_cfg = load_cfg(opt.pose_cfg)["infer"]
    track_cfg = load_cfg(opt.track_cfg)["infer"]
    dataset_dir = opt.dataset_dir
    dataset = opt.dataset
    batch_size = opt.batch_size

    data_paths = parse_dataset_dir(dataset_dir=dataset_dir, dataset=dataset)

    det_model = YOLOv7(det_cfg, device)
    print(f"Object detection model({det_cfg['model_name']}) is loaded.")
    if pose_cfg["model_name"] == "alphapose":
        pose_model = AlphaPose(pose_cfg, device)
    else:
        pose_model = SimpleHRNetWrapper(pose_cfg, device)

    print(f"Pose estimation model({pose_cfg['model_name']}) is loaded.")

    for dataset_type in ["train", "test"]:
        video_dir = data_paths.get(dataset_type)
        frame_dir = data_paths.get(f"{dataset_type}_frames")

        if video_dir and frame_dir:
            video_list = parse_video_list(dataset_dir, dataset, dataset_type)
            for video_path in tqdm(video_list, desc=f"Processing {dataset_type} videos"):
                track_model = BYTETracker(track_cfg)
                video_name = video_path.stem
                video_output_dir = os.path.join(frame_dir, video_name)
                extract_and_process_frames(video_path, video_output_dir, video_name, det_model, track_model, pose_model, batch_size)
