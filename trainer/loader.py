
__all__ = ['DEFAULT_DATA_PATH', 'DLDataset', 'ValMetricDLDataset']

from torch.utils.data import Dataset
import torch
import os
from torchvision import transforms
import numpy as np


class DLDataset(Dataset):
    def __init__(self, root, mode, unlabeled=False, use_gt_data=False, pre_seq_len=11, aft_seq_len=11, ep_len=22):
        if use_gt_data:
            self.mask_path = os.path.join(root, f"{mode}_gt_masks.pt")
        else:
            self.mask_path = os.path.join(root, f"{mode}_masks.pt")
            
        self.mode = mode
        print("INFO: Loading masks from", self.mask_path)
        if unlabeled:
            print("INFO: Using unlabeled masks for training!")
            # Load the first numpy file
            mask1 = torch.load(self.mask_path)
            # Load the second numpy file and squeeze if necessary
            mask2 = torch.from_numpy(np.load(os.path.join(root, "unlabeled_masks.npy")))
            print(mask2.shape)
            sys.exit()
            # Concatenate the tensors along dim=0
            self.masks = torch.cat([mask1, mask2], dim=0)
        else:
            self.masks = torch.load(self.mask_path)
        print("INFO: The number of masks:",  self.masks.shape[0])
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.pre_seq_len=pre_seq_len
        self.aft_seq_len=aft_seq_len
        self.seq_per_ep = ep_len - (pre_seq_len + aft_seq_len) + 1

    def __len__(self):
        return self.masks.shape[0] * self.seq_per_ep
    
    def __getitem__(self, idx):
        ep_idx = idx // self.seq_per_ep
        offset = idx % self.seq_per_ep
        total_len = self.pre_seq_len + self.aft_seq_len
        
        if self.mode == "train":
            ep = self.transform(self.masks[ep_idx, offset:offset+total_len])
        else:
            ep = self.masks[ep_idx, offset:offset+total_len]
        data = ep[:self.pre_seq_len].long()
        labels = ep[self.pre_seq_len:].long()
        return data, labels

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
