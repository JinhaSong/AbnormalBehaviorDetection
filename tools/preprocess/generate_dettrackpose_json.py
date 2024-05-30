import os
import sys
import cv2
import json
import shutil
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.cfg import load_cfg
from util.dataset import parse_dataset_dir, parse_video_list
from lib.yolo.yolov7.det import YOLOv7
from lib.pose.alphapose.pose import AlphaPose
from lib.tracker.bytetracker.BYTETracker import BYTETracker


def save_results(frame, frame_output_dir, frame_name, pose_object_result, video_name, frame_number):
    os.makedirs(frame_output_dir, exist_ok=True)
    frame_path = os.path.join(frame_output_dir, f"{frame_name}.jpg")
    cv2.imwrite(frame_path, frame)

    formatted_result = {
        "video_name": video_name,
        "frame_number": frame_number,
        "result": pose_object_result
    }

    json_path = os.path.join(frame_output_dir, f"{frame_name}.json")
    with open(json_path, "w") as json_file:
        json.dump(formatted_result, json_file, indent=4)


def process_frame(frame, det_model, track_model, pose_model):
    detection_result = det_model.inference_image_batch([frame])[0]
    pose_object_result = pose_model.inference_image(frame, detection_result)
    tracking_results = track_model.update(pose_object_result)
    return tracking_results


def extract_and_process_frames(video_path, output_dir, frame_prefix, det_model, track_model, pose_model):
    os.makedirs(output_dir, exist_ok=True)
    video_path_str = str(video_path)
    vidcap = cv2.VideoCapture(video_path_str)
    if not vidcap.isOpened():
        print(f"Error: Unable to open video file {video_path_str}")
        return

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc=f"Processing frames from {video_path.stem}") as pbar:
        success, frame = vidcap.read()
        count = 0
        while success:
            frame_name = f"{frame_prefix}_{count:05d}"
            pose_object_result = process_frame(frame, det_model, track_model, pose_model)
            save_results(frame, output_dir, frame_name, pose_object_result, frame_prefix, count)
            success, frame = vidcap.read()
            count += 1
            pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--device", type=str, default="0", help="GPU device number")
    parser.add_argument("--det-cfg", type=str, default="cfgs/cfg_yolov7x.yml", help="Path of object detection parameter")
    parser.add_argument("--pose-cfg", type=str, default="cfgs/cfg_alphapose.yml", help="Path of pose estimation parameter")
    parser.add_argument("--track-cfg", type=str, default="cfgs/cfg_bytetrack.yml", help="Path of object tracking parameter")
    parser.add_argument("--dataset-dir", required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset", choices=["cuhk", "shanghaitech", "ubnormal", "ucf"], required=True, help="Type of the dataset")

    opt = parser.parse_known_args()[0]

    device = opt.device
    det_cfg = load_cfg(opt.det_cfg)["infer"]
    pose_cfg = load_cfg(opt.pose_cfg)["infer"]
    track_cfg = load_cfg(opt.track_cfg)["infer"]
    dataset_dir = opt.dataset_dir
    dataset = opt.dataset

    data_paths = parse_dataset_dir(dataset_dir=dataset_dir, dataset=dataset)

    det_model = YOLOv7(det_cfg, device)
    print(f"Object detection model({det_cfg['model_name']}) is loaded.")
    pose_model = AlphaPose(pose_cfg, device)
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
                extract_and_process_frames(video_path, video_output_dir, video_name, det_model, track_model, pose_model)

