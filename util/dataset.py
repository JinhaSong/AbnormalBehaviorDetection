import os
from pathlib import Path


def parse_dataset_dir(dataset_dir, dataset):
    if dataset == "cuhk":
        training_videos_dir = os.path.join(dataset_dir, "training", "training_videos")
        training_frames_dir = os.path.join(dataset_dir, "training", "training_videos_frame")
        testing_videos_dir = os.path.join(dataset_dir, "testing", "testing_videos")
        testing_frames_dir = os.path.join(dataset_dir, "testing", "testing_videos_frame")
        return {
            "train": training_videos_dir,
            "train_frames": training_frames_dir,
            "test": testing_videos_dir,
            "test_frames": testing_frames_dir
        }

    elif dataset == "shanghaitech":
        training_videos_dir = os.path.join(dataset_dir, "training", "videos")
        training_frames_dir = os.path.join(dataset_dir, "training", "frames")
        return {
            "train": training_videos_dir,
            "train_frames": training_frames_dir,
            "test": None,
            "test_frames": None
        }

    elif dataset == "ucf":
        videos_dir = os.path.join(dataset_dir, "Videos")
        frames_dir = os.path.join(dataset_dir, "frames")
        return {
            "train": videos_dir,
            "train_frames": frames_dir,
            "test": None,
            "test_frames": None
        }
    elif dataset == "ubnormal":
        videos_dir = os.path.join(dataset_dir, "videos")
        frames_dir = os.path.join(dataset_dir, "frames")
        return {
            "train": videos_dir,
            "train_frames": frames_dir,
            "test": None,
            "test_frames": None
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def parse_video_list(dataset_dir, dataset, dataset_type):
    video_paths = []

    if dataset == "cuhk":
        if dataset_type == "train":
            videos_dir = os.path.join(dataset_dir, "training", "training_videos")
        elif dataset_type == "test":
            videos_dir = os.path.join(dataset_dir, "testing", "testing_videos")
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    elif dataset == "shanghaitech":
        if dataset_type == "train":
            videos_dir = os.path.join(dataset_dir, "training", "videos")
        elif dataset_type == "test":
            raise ValueError(f"ShanghaiTech dataset does not have test videos in the given structure")
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    elif dataset == "ucf":
        videos_dir = os.path.join(dataset_dir, "Videos")

    elif dataset == "ubnormal":
        if dataset_type != "train":
            raise ValueError(f"UBnormal dataset only supports 'train' dataset type")
        videos_dir = os.path.join(dataset_dir, "videos")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    video_dir_path = Path(videos_dir)

    for ext in ["*.avi", "*.mp4"]:
        video_paths.extend(sorted(video_dir_path.rglob(ext)))

    if not video_paths:
        raise ValueError(f"No video files found in directory: {videos_dir}")

    return video_paths