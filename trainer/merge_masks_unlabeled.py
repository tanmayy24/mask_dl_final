import argparse
import os
import tqdm
import numpy as np

def main(args):
    all_paths = []
    range_start = 2000 
    range_end = 14999
    for i in tqdm.tqdm(range(range_start, range_end)):
        file_path = os.path.join(args.data_root, f"video_{i:05d}_mask.npy")
        if os.path.exists(file_path):  # Check if the file exists
            all_paths.append(file_path)
            print(f"Added path for video {i}: {file_path}")

    # Save the list of paths to a .npy file
    np.save(args.output_file, np.array(all_paths))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/scratch/rn2214/labeled/")
    parser.add_argument(
        "--output_file", type=str, default="unlabeled_masks.npy"
    )

    args = parser.parse_args()

    main(args)
