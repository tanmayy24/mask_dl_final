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
    "ckpt_path" : "/scratch/tk3309/mask_dl_final/slurm/checkpoints_finetune/method=SS_simvp=simvp_epoch=18-val_loss=0.014.ckpt_inc_every_n_epoch=20_max_sample_steps=5_schedule_k=1.05_unlabeled=False_use_gt_data=False_schedule_type=exponential/simvp_ss_epoch=6-valid_last_frame_iou=0.458.ckpt",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
jaccard_index = JaccardIndex(task='multiclass', num_classes=49)
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
    
class HiddenDLDataset(Dataset):
    """ Dataset for handling the loading of mask data for training or validation. """
    def __init__(self, root, pre_seq_len=11):
        mask_type = "hidden_masks.pt"
        self.mask_path = os.path.join(root, f"{mask_type}")
        print(f"INFO: Loading masks from {self.mask_path}")
        self.masks = torch.load(self.mask_path)
        self.pre_seq_len = pre_seq_len

    def __len__(self):
        return self.masks.shape[0]

    def __getitem__(self, idx):
        episode = self.masks[idx]
        data = episode[:self.pre_seq_len].long()
        return data

class MetricDLDataset(Dataset):
    def __init__(self, root, set_to_predict):
        self.x_dataset = DLDataset(root, set_to_predict, use_gt_data=True)
        self.y_dataset = DLDataset(root, set_to_predict, use_gt_data=True)
    
    def __len__(self):
        return len(self.x_dataset)

    def __getitem__(self, idx):
        x, _ = self.x_dataset[idx]
        _, y = self.y_dataset[idx]
        return x, y
    
def evaluate_model(data_loader, model, device):
    all_yhat = []
    all_targets = []
    print("INFO: Starting validation model evaluation...")
    
    for inputs, targets in tqdm(data_loader, desc="Evaluating model"):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            y_hat = model.sample_autoregressive(inputs, 11)
        all_yhat.append(y_hat[:, -1].cpu())
        all_targets.append(targets[:, -1].cpu())

    all_yhat_tensor = torch.cat(all_yhat)
    all_targets_tensor = torch.cat(all_targets)
    return all_yhat_tensor, all_targets_tensor

def main(config):
    set_to_predict = "val"
    module = MaskSimVPScheduledSamplingModule.load_from_checkpoint(
        config["ckpt_path"], data_root=config["data_root"], use_gt_data=True, unlabeled=False, load_datasets=False
    )
    dataset = MetricDLDataset(config["data_root"], set_to_predict)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=1, shuffle=False, pin_memory=True)
    predictions, targets = evaluate_model(data_loader, module, config["device"])
    torch.save(predictions, "val_preds_finetune.pt")
    print(f"The shape of predictions:", predictions.shape)
    print("INFO: Predictions saved to 'val_preds_finetune.pt'.")
    print(f"The final validation IoU: {jaccard_index(predictions, targets)}")
    
def predict_hidden(config):
    module = MaskSimVPScheduledSamplingModule.load_from_checkpoint(
        config["ckpt_path"], data_root=config["data_root"], use_gt_data=True, unlabeled=False, load_datasets=False
    )
    hidden_dataset = HiddenDLDataset(config["data_root"])
    hidden_data_loader = DataLoader(hidden_dataset, batch_size=32, num_workers=1, shuffle=False, pin_memory=True)
    all_yhat = []
    print("INFO: Starting model evaluation...for hidden dataset")
    for inputs in tqdm(hidden_data_loader, desc="Hidden Prediction model"):
        inputs = inputs.to(config["device"])
        with torch.no_grad():
            y_hat = module.sample_autoregressive(inputs, 11)
        all_yhat.append(y_hat[:, -1].cpu())
    all_yhat_tensor = torch.cat(all_yhat)
    print(f"The shape of predictions:", all_yhat_tensor.shape)
    torch.save(all_yhat_tensor, "hidden_preds_finetune.pt")
    print("INFO: Predictions saved to 'hidden_preds_finetune.pt'.")

if __name__ == "__main__":
    main(config)
    #predict_hidden(config)
