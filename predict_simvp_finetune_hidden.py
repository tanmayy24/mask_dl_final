import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchmetrics import JaccardIndex
from lightning import seed_everything
from trainer.config import DEFAULT_DATA_PATH, SEED
from trainer.trainer_finetune import MaskSimVPScheduledSamplingModule
import numpy as np

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

class HiddenDLDataset(Dataset):
    """ Dataset for handling the loading of mask data for training or validation. """
    def __init__(self, root, pre_seq_len=11):

        self.root = root
        self.pre_seq_len = pre_seq_len

        self.range_start = 15000
        self.range_end = 19999

        self.pre_seq_len = pre_seq_len

    def __len__(self):
        return self.range_end - self.range_start+1

    def __getitem__(self, idx):
        actual_idx = self.range_start + idx
        mask_path = os.path.join(self.root, f"video_{actual_idx:05d}_mask.npy")
        file_name = f"video_{actual_idx:05d}_mask.npy"
        mask = np.load(mask_path).astype(np.int64)
        data = mask[:self.pre_seq_len]
        print(f"Loaded mask for video {actual_idx}: shape {data.shape}")
        return file_name, torch.from_numpy(data)

def predict_hidden(config):
    model = MaskSimVPScheduledSamplingModule.load_from_checkpoint(
        config["ckpt_path"], data_root=config["data_root"], use_gt_data=True, unlabeled=False, load_datasets=False
    )
    hidden_dataset = HiddenDLDataset("/scratch/rn2214/labeled/hidden/")
    hidden_data_loader = DataLoader(hidden_dataset, batch_size=64, num_workers=1, shuffle=False, pin_memory=True)
    all_yhat = []
    print("INFO: Starting model predictions...for hidden dataset")
    for file_name, inputs in tqdm(hidden_data_loader, desc="Hidden Prediction model"):
        print(f"The filename is:", file_name)
        inputs = inputs.to(config["device"])
        with torch.no_grad():
            y_hat = model.sample_autoregressive(inputs, 11)
        all_yhat.append(y_hat[:, -1].cpu())
    all_yhat_tensor = torch.cat(all_yhat)
    print(f"The shape of predictions:", all_yhat_tensor.shape)
    torch.save(all_yhat_tensor, "hidden_preds_team3_final.pt")
    print("INFO: Predictions saved to hidden_preds_team3_final.pt")

if __name__ == "__main__":
    predict_hidden(config)
