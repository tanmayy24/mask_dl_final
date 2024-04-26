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

from trainer.trainer import MaskSimVPModule
seed_everything(0)
torch.backends.cudnn.deterministic = True

data_root  = "/scratch/tk3309/dl_data/dataset/"
ckpt_path = "/scratch/tk3309/mask_dl_final/slurm/checkpoints/in_shape=11-49-160-240_hid_S=64_hid_T=512_N_S=4_N_T=8_model_type=gSTA_batch_size=4_lr=0.001_weight_decay=0.0_max_epochs=20_pre_seq_len=11_aft_seq_len=1_unlabeled=False_downsample=True/simvp_epoch=19-val_loss=0.017-v1.ckpt"
module = MaskSimVPModule.load_from_checkpoint(ckpt_path, data_root=data_root,use_gt_data=True, unlabeled=False, load_datasets=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

class DLDataset(Dataset):
    def __init__(self, root, mode, use_gt_data=False, pre_seq_len=11, aft_seq_len=11):
        if use_gt_data:
            self.mask_path = os.path.join(root, f"{mode}_gt_masks.pt")
        else:
            self.mask_path = os.path.join(root, f"{mode}_masks.pt")
        self.mode = mode
        print("INFO: Loading masks from", self.mask_path)
        self.masks = torch.load(self.mask_path)
        self.pre_seq_len=pre_seq_len
        self.aft_seq_len=aft_seq_len

    def __len__(self):
        return self.masks.shape[0]
    
    def __getitem__(self, idx):
        ep = self.masks[idx]
        data = ep[:self.pre_seq_len].long()
        labels = ep[self.pre_seq_len:].long()
        return data, labels


dataset = DLDataset(data_root, "val", use_gt_data=True, pre_seq_len=11, aft_seq_len=1)
data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, 
            num_workers=1, shuffle=False, pin_memory=True
        )

all_yhat = []
all_targets = []
jaccard = JaccardIndex(task='multiclass', num_classes=49)
for inputs, targets in tqdm.tqdm(data_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        y_hat = module.sample_autoregressive(inputs, 11)
    all_yhat.append(y_hat[:, -1].to("cpu"))
    all_targets.append(targets[:, -1].to("cpu"))

all_targets_tensor = torch.cat(all_targets, dim=0)
all_yhat_tensor = torch.cat(all_yhat, dim=0)

print(all_targets_tensor.shape)
print(all_yhat_tensor.shape)
#torch.save(all_yhat_tensor, "val_preds.pt")
print(f"The final IoU: {jaccard(all_yhat_tensor, all_targets_tensor)}")