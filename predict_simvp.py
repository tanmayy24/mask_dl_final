import torch 
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os 
from torchvision import transforms
import tqdm
import torch

from torchmetrics import JaccardIndex
from lightning import seed_everything

from maskpredformer.trainer import MaskSimVPModule
seed_everything(0)
torch.backends.cudnn.deterministic = True

data_root  = "/scratch/tk3309/dl_data/dataset/"
ckpt_path = "/scratch/tk3309/simvp_epoch=16-val_loss=0.014.ckpt"
module = MaskSimVPModule.load_from_checkpoint(ckpt_path, data_root=data_root,use_gt_data=True, unlabeled=False, load_datasets=False)


device = "cuda" if torch.cuda.is_available() else "cpu"

class DLDataset(Dataset):
    def __init__(self, root, mode, unlabeled=False, use_gt_data=False, pre_seq_len=11, aft_seq_len=11, ep_len=22):
        if use_gt_data:
            self.mask_path = os.path.join(root, f"{mode}_gt_masks.pt")
        else:
            self.mask_path = os.path.join(root, f"{mode}_masks.pt")
            
        self.mode = mode
        print("INFO: Loading masks from", self.mask_path)
        if unlabeled:
            self.masks = torch.cat([
                torch.load(self.mask_path), 
                torch.load(os.path.join(root, f"unlabeled_masks.pt")).squeeze()
            ], dim=0)
        else:
            self.masks = torch.load(self.mask_path)
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
    


@torch.no_grad()
def get_predictions(module, x):
    x = x.unsqueeze(0).to(module.device)
    cur_seq = module.sample_autoregressive(x, 11)
    y_hat = cur_seq.squeeze(0).cpu().type(torch.uint8)
    return y_hat

dataset = DLDataset(data_root, "val", use_gt_data=True, pre_seq_len=11, aft_seq_len=1)
data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, 
            num_workers=1, shuffle=False, pin_memory=True
        )

iou_list = []
jaccard = JaccardIndex(task='multiclass', num_classes=49)
# Iterate over the data loader
for inputs, targets in tqdm.tqdm(data_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    # Compute predictions
    with torch.no_grad():
        y_hat = module.sample_autoregressive(inputs, 11)
    # Append predictions and targets to lists
    iou_score = jaccard(y_hat[:, -1].to("cpu"), targets[:, -1].to("cpu"))
    iou_list.append(iou_score.item())

# Concatenate lists to tensors
mean_iou = np.mean(iou_list)
print(f"The final Jaccard Index: {mean_iou}")