import os
import random
import argparse


def create_train_test_lists(data_path, train_ratio=0.6, output_path='.'):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    # List all video folders in train and test directories
    train_videos = [os.path.join("train", f) for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]
    test_videos = [os.path.join("test", f) for f in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, f))]

    # Shuffle the lists
    random.shuffle(train_videos)
    random.shuffle(test_videos)

    # Split test videos into normal and anomaly
    normal_videos = train_videos
    anomaly_videos = test_videos

    # Calculate number of videos for training and testing
    num_normal_train = int(len(normal_videos) * train_ratio)
    num_anomaly_train = int(len(anomaly_videos) * train_ratio)

    # Split normal and anomaly videos into train and test sets
    normal_train = normal_videos[:num_normal_train]
    normal_test = normal_videos[num_normal_train:]
    anomaly_train = anomaly_videos[:num_anomaly_train]
    anomaly_test = anomaly_videos[num_anomaly_train:]

    # Combine normal and anomaly videos for train and test sets
    train_list = [video for video in normal_train + anomaly_train]
    test_list = [video for video in normal_test + anomaly_test]

    # Shuffle the train and test lists
    random.shuffle(train_list)
    random.shuffle(test_list)

    # Write train.list and test.list files
    with open(os.path.join(output_path, 'train.list'), 'w') as train_file:
        for video in train_list:
            train_file.write(f"{video}\n")

    with open(os.path.join(output_path, 'test.list'), 'w') as test_file:
        for video in test_list:
            test_file.write(f"{video}\n")

    print(f"train.list and test.list created with {len(train_list)} and {len(test_list)} videos respectively.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train and test lists for video anomaly detection.")
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help="Path to the dataset directory containing 'train' and 'test' folders.")
    parser.add_argument('--train-ratio', type=float, default=0.6, help="Ratio of training data (default: 0.6).")
    parser.add_argument('--output-dir', type=str, default='.',
                        help="Path to save the output 'train.list' and 'test.list' files (default: current directory).")

    args = parser.parse_args()

    create_train_test_lists(args.dataset_dir, args.train_ratio, args.output_dir)