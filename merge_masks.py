import argparse
import torch
import os
import numpy as np
import tqdm


def main(args):
    all_masks = []
    range_start = 0 if args.split == "train" else 1000
    range_end = 1000 if args.split == "train" else 2000
    for i in tqdm.tqdm(range(range_start, range_end)):
        mask = np.load(
            os.path.join(args.data_root, args.split, f"video_{i}", "mask.npy")
        )
        all_masks.append(mask)

    all_masks = np.stack(all_masks)
    torch.save(torch.from_numpy(all_masks), args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data/Dataset_Student")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--output_file", type=str, default="data/DL/train_gt_masks_new.pt"
    )

    args = parser.parse_args()

    main(args)
