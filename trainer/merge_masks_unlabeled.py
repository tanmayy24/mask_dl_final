import argparse
import os
import numpy as np
import tqdm

def main(args):
    all_masks = []
    range_start = 2000 
    range_end = 14999
    for i in tqdm.tqdm(range(range_start, range_end)):
        mask_path = os.path.join(args.data_root, f"video_{i:05d}_mask.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            print(f"Loaded mask shape for video {i}: {mask.shape}")
            all_masks.append(mask)
        else:
            print(f"File not found: {mask_path}")

    all_masks = np.stack(all_masks)
    np.save(args.output_file, all_masks)  # Save as a NumPy array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/scratch/rn2214/labeled/")
    parser.add_argument("--output_file", type=str, default="unlabeled_masks.npy")

    args = parser.parse_args()

    main(args)
