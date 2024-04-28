import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchmetrics import JaccardIndex
from lightning import seed_everything
from trainer.config import DEFAULT_DATA_PATH, SEED
from trainer.trainer_finetune import MaskSimVPScheduledSamplingModule

# Set up seeds and GPU options
seed_everything(SEED)
torch.backends.cudnn.deterministic = True

# Configuration settings
config = {
    "data_root": DEFAULT_DATA_PATH,
    "ckpt_path" : "/scratch/tk3309/mask_dl_final/slurm/checkpoints_finetune/method=SS_simvp=simvp_epoch=19-val_loss=0.017-v1.ckpt_inc_every_n_epoch=20_max_sample_steps=5_schedule_k=1.05_unlabeled=False_use_gt_data=False_schedule_type=exponential/simvp_ss_epoch=39-valid_last_frame_iou=0.402.ckpt",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class DLDataset(Dataset):
    """ Dataset for handling the loading of mask data for training or validation. """
    def __init__(self, root, mode, use_gt_data=False, pre_seq_len=11, aft_seq_len=1):
        mask_type = "gt_masks.pt" if use_gt_data else "masks.pt"
        self.mask_path = os.path.join(root, f"{mode}_{mask_type}")
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
    jaccard_index = JaccardIndex(task='multiclass', num_classes=49)
    for inputs, targets in tqdm(data_loader, desc="Evaluating model"):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            y_hat = model.sample_autoregressive(inputs, 11)
        all_yhat.append(y_hat[:, -1].cpu())
        all_targets.append(targets[:, -1].cpu())

    all_yhat_tensor = torch.cat(all_yhat)
    all_targets_tensor = torch.cat(all_targets)
    return all_yhat_tensor, all_targets_tensor, jaccard_index

def main(config):
    set_to_predict = "val"
    module = MaskSimVPScheduledSamplingModule.load_from_checkpoint(
        config["ckpt_path"], data_root=config["data_root"], use_gt_data=True, unlabeled=False, load_datasets=False
    )
    dataset = DLDataset(config["data_root"],set_to_predict, use_gt_data=True)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=1, shuffle=False, pin_memory=True)
    predictions, targets, jaccard = evaluate_model(data_loader, module, config["device"])
    torch.save(predictions, "val_preds_finetune.pt")
    print("INFO: Predictions saved to 'val_preds_finetune.pt'.")
    print(f"The final IoU: {jaccard(predictions, targets)}")

if __name__ == "__main__":
    main(config)
