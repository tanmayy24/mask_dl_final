import argparse
import torch
import os
import numpy as np
import tqdm


def main(args):
    all_masks = []
    range_start = 2000 
    range_end = 14999
    for i in tqdm.tqdm(range(range_start, range_end)):
        mask = np.load(
            os.path.join(args.data_root, f"video_{i:05d}_mask.npy")
        )
        print(f"Loaded mask before shape for video {i}: {mask.shape}")
        all_masks.append(mask)
        print(f"Loaded mask after shape for video {i}: {mask.shape}")

    all_masks = np.stack(all_masks)
    torch.save(torch.from_numpy(all_masks), args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/scratch/rn2214/labeled/")
    parser.add_argument("--split", type=str, default="unlabeled")
    parser.add_argument(
        "--output_file", type=str, default="unlabeled_masks.pt"
    )

    args = parser.parse_args()

    main(args)
