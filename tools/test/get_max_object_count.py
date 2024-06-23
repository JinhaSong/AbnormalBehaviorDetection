import os
import json
import argparse
from tqdm import tqdm

def count_max_objects(directory):
    max_objects = 0

    json_files = [os.path.join(root, file)
                  for root, _, files in os.walk(directory)
                  for file in files if file.endswith('.json')]

    for json_path in tqdm(json_files, desc="Processing JSON files"):
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        try:
            num_objects = len(json_data['result'])
            if num_objects > max_objects:
                max_objects = num_objects
        except:
            print(json_path)

    return max_objects

def main():
    parser = argparse.ArgumentParser(description="Count the maximum number of objects in frames.")
    parser.add_argument('--input-dir', type=str, required=True, help='Path to the input dataset directory containing frames and JSON files.')
    args = parser.parse_args()

    max_objects = count_max_objects(args.input_dir)
    print(f"The maximum number of objects in any frame is: {max_objects}")

if __name__ == "__main__":
    main()