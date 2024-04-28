import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchmetrics import JaccardIndex
from lightning import seed_everything
from trainer.config import DEFAULT_DATA_PATH, SEED
from trainer.trainer import MaskSimVPModule

# Setting seeds and deterministic behavior
seed_everything(SEED)
torch.backends.cudnn.deterministic = True

# Configuration settings
data_root = DEFAULT_DATA_PATH
ckpt_path = "/scratch/tk3309/mask_dl_final/slurm/checkpoints/in_shape=11-49-160-240_hid_S=64_hid_T=512_N_S=4_N_T=8_model_type=gSTA_batch_size=16_lr=0.001_weight_decay=0.0_max_epochs=20_pre_seq_len=11_aft_seq_len=1_unlabeled=True_downsample=True/simvp_epoch=18-val_loss=0.014.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"
class DLDataset(Dataset):
    """ Dataset for loading masks, either ground truth or generated. """
    def __init__(self, root, mode, use_gt_data=False, pre_seq_len=11, aft_seq_len=1):
        self.mask_path = os.path.join(root, f"{mode}_" + ("gt_masks.pt" if use_gt_data else "masks.pt"))
        print(f"INFO: Loading masks from {self.mask_path}")
        self.masks = torch.load(self.mask_path)
        self.pre_seq_len = pre_seq_len
        self.aft_seq_len = aft_seq_len

    def __len__(self):
        return self.masks.shape[0]

    def __getitem__(self, idx):
        episode = self.masks[idx]
        data = episode[:self.pre_seq_len].long()
        labels = episode[self.pre_seq_len:].long()
        return data, labels

def evaluate_model(data_loader, model, device):
    all_yhat = []
    all_targets = []
    print("INFO: Starting model evaluation...")
    jaccard = JaccardIndex(task='multiclass', num_classes=49)
    for inputs, targets in tqdm(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            y_hat = model.sample_autoregressive(inputs, 11)
        all_yhat.append(y_hat[:, -1].cpu())
        all_targets.append(targets[:, -1].cpu())
    return torch.cat(all_yhat), torch.cat(all_targets), jaccard

def main():
    set_to_predict = "val"
    module = MaskSimVPModule.load_from_checkpoint(ckpt_path, data_root=data_root, use_gt_data=True, unlabeled=False, load_datasets=False)
    dataset = DLDataset(data_root, set_to_predict, use_gt_data=True)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=1, shuffle=False, pin_memory=True)
    all_yhat_tensor, all_targets_tensor, jaccard = evaluate_model(data_loader, module, device)
    torch.save(all_yhat_tensor, "val_preds.pt")
    print("INFO: Predictions saved to 'val_preds.pt'.")
    print(f"The final IoU: {jaccard(all_yhat_tensor, all_targets_tensor)}")

if __name__ == "__main__":
    main()
