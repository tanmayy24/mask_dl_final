import argparse
import os
import numpy as np
import tqdm

def process_batch(batch, output_file, mode='a'):
    stacked_batch = np.stack(batch)
    np.save(output_file, stacked_batch, allow_pickle=False, fix_imports=False)
    if mode == 'a':
        with open(output_file, 'ab') as f:  # Append mode for file
            np.save(f, stacked_batch)

def main(args):
    all_masks = []
    batch_size = 100  # Adjust batch size based on your memory capacity
    current_batch = []

    range_start = 2000
    range_end = 14999
    for i in tqdm.tqdm(range(range_start, range_end)):
        mask_path = os.path.join(args.data_root, f"video_{i:05d}_mask.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            print(f"Loaded mask shape for video {i}: {mask.shape}")
            current_batch.append(mask)
            if len(current_batch) >= batch_size:
                process_batch(current_batch, args.output_file)
                current_batch = []  # Clear the current batch
        else:
            print(f"File not found: {mask_path}")

    # Process any remaining masks
    if current_batch:
        process_batch(current_batch, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/scratch/rn2214/labeled/")
    parser.add_argument("--output_file", type=str, default="unlabeled_masks.npy")

    args = parser.parse_args()

    main(args)
