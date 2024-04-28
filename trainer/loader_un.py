
__all__ = ['DEFAULT_DATA_PATH', 'DLDataset', 'ValMetricDLDataset']

from torch.utils.data import Dataset
import torch
import os
from torchvision import transforms
import numpy as np


class DLDataset(Dataset):
    def __init__(self, root, mode, unlabeled=False, use_gt_data=False, pre_seq_len=11, aft_seq_len=11, ep_len=22):
        self.mask_path = os.path.join(root, f"{mode}_gt_masks.pt")
        self.mode = mode
        print("INFO: Loading masks from", self.mask_path)
        self.unlabeled = unlabeled
        if unlabeled:
            print("INFO: Using unlabeled masks for training!")
            self.mask1 = torch.load(self.mask_path)
            self.mask2 = np.load(os.path.join(root, "unlabeled_path.npy"))
            print("INFO: The number of unlabeled masks:",  self.mask2.size)
            self.masks = self.mask1.shape[0]+self.mask2.size
        else:
            self.masks = torch.load(self.mask_path)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.pre_seq_len=pre_seq_len
        self.aft_seq_len=aft_seq_len
        self.seq_per_ep = ep_len - (pre_seq_len + aft_seq_len) + 1

    def __len__(self):
        if(self.mode == "train" and self.unlabeled):
            print("INFO: The number of total masks:",  self.masks)
            return self.masks* self.seq_per_ep
        else:
            print("INFO: The number of total masks:",  self.masks.shape[0])
            return self.masks.shape[0] * self.seq_per_ep
    
    def __getitem__(self, idx):
        episode_index = idx // self.seq_per_ep
        sequence_offset = idx % self.seq_per_ep
        total_length = self.pre_seq_len + self.aft_seq_len
        if self.mode == "train":
            if self.unlabeled:
                if 0 <= episode_index <= 999:
                    episode_data = self.transform(self.mask1[episode_index, sequence_offset:sequence_offset+total_length])
                else:
                    try:
                        path_to_load = self.mask2[episode_index-1000]
                        npy_data = np.load(path_to_load)
                        loaded_mask = torch.from_numpy(npy_data)
                        episode_data = self.transform(loaded_mask[sequence_offset:sequence_offset+total_length])
                    except FileNotFoundError:
                        raise Exception(f"Mask file not found: {path_to_load}")
            else:
                episode_data = self.transform(self.masks[episode_index, sequence_offset:sequence_offset+total_length])
        else:
            episode_data = self.masks[episode_index, sequence_offset:sequence_offset+total_length]
        input_data = episode_data[:self.pre_seq_len].long()
        labels = episode_data[self.pre_seq_len:].long()
        return input_data, labels

class ValMetricDLDataset(Dataset):
    def __init__(self, root):
        self.val_x_dataset = DLDataset(root, "val", use_gt_data=True)
        self.val_y_dataset = DLDataset(root, "val", use_gt_data=True)
    
    def __len__(self):
        return len(self.val_x_dataset)

    def __getitem__(self, idx):
        x, _ = self.val_x_dataset[idx]
        _, y = self.val_y_dataset[idx]
        return x, y