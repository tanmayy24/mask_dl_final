
__all__ = ['inv_sigmoid_schedule', 'exp_schedule', 'MaskSimVPScheduledSamplingModule']
import torch
import os

import lightning as pl
from torchmetrics import JaccardIndex

import wandb
import math
import random
import numpy as np

from .simvp_model import MaskSimVP
from .loader import DLDataset, ValMetricDLDataset
def inv_sigmoid_schedule(x, n, k):
    y = k / (k+math.exp(((x-(n//2))/(n//20))/k))
    return y

def exp_schedule(x, n, k=np.e):
    t = 100 * np.maximum((x / n)-0.033,0)
    return k ** -t
class MaskSimVPScheduledSamplingModule(pl.LightningModule):
    def __init__(self, 
                 in_shape, hid_S, hid_T, N_S, N_T, model_type,
                 batch_size, lr, weight_decay, max_epochs, data_root, use_gt_data,
                 sample_step_inc_every_n_epoch, schedule_k=1.05, max_sample_steps=5,
                 schedule_type="exponential", load_datasets=True,
                 pre_seq_len=11, aft_seq_len=1, drop_path=0.0, unlabeled=False, downsample=False,
                ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MaskSimVP(
            in_shape, hid_S, hid_T, N_S, N_T, model_type, downsample=downsample, drop_path=drop_path,
            pre_seq_len=pre_seq_len, aft_seq_len=aft_seq_len
        )
        if load_datasets:
            self.train_set = DLDataset(data_root, "train", unlabeled=unlabeled, use_gt_data=True, pre_seq_len=pre_seq_len, aft_seq_len=max_sample_steps+1)
            self.val_set = ValMetricDLDataset(data_root)
            self.schedule_max = (len(self.train_set)//batch_size) * sample_step_inc_every_n_epoch
            print(f"Schedule max: {self.schedule_max}")
        else:
            self.schedule_max = -1 # dummy value
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.iou_metric = JaccardIndex(task='multiclass', num_classes=49)
        
        self.schedule_idx = 0
        self.sample_steps = 1
        self.sampled_count = 0

    def get_p(self):
        if self.hparams.schedule_type == "exponential":
            p = 1-exp_schedule(self.schedule_idx, self.schedule_max, self.hparams.schedule_k)
        elif self.hparams.schedule_type == "inverse_sigmoid":
            p = 1 - inv_sigmoid_schedule(self.schedule_idx, self.schedule_max, self.hparams.schedule_k)
        else:
            raise NotImplementedError(f"Schedule type {self.hparams.schedule_type} not implemented")
        return p

    def sample_or_not(self):
        assert self.schedule_idx < self.schedule_max, "Schedule idx larger than max, something wrong with schedule"
        p = self.get_p()
        self.schedule_idx += 1
        return random.random() < p
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, 
            num_workers=8, shuffle=True, pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set, batch_size=self.hparams.batch_size, 
            num_workers=8, shuffle=False, pin_memory=True
        )

    @torch.no_grad()
    def sample_autoregressive(self, x, t):
        cur_seq = x.clone()
        for _ in range(t):
            y_hat_logit_t = self.model(cur_seq)
            y_hat = torch.argmax(y_hat_logit_t, dim=2) # get current prediction
            cur_seq = torch.cat([cur_seq[:, 1:], y_hat], dim=1) # autoregressive concatenation
        return cur_seq
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.sample_or_not():
            self.sampled_count += 1
            x = self.sample_autoregressive(x, self.sample_steps)
            y = y[:, self.sample_steps:self.sample_steps+1] # get the next label after sampling model `sample_steps` times
        else:
            # no change in x
            y = y[:, 0:1] # get the normal training label
        
        y_hat_logits = self.model(x)
        
        # Flatten batch and time dimensions
        b, t, *_ = y_hat_logits.shape
        y_hat_logits = y_hat_logits.view(b*t, *y_hat_logits.shape[2:])
        y = y.view(b*t, *y.shape[2:])

        loss = self.criterion(y_hat_logits, y)
        
        self.log("train_loss", loss)
        if self.logger:
            self.logger.log_metrics(
                {
                    "sample_steps": self.sample_steps,
                    "schedule_idx": self.schedule_idx,
                    "schedule_prob": self.get_p(),
                    "sampled_count": self.sampled_count,
                }
            )
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.sample_autoregressive(x, 11)
        iou = self.iou_metric(y_hat[:, -1], y[:, -1])
        self.log("valid_last_frame_iou", self.iou_metric, on_step=False, on_epoch=True, sync_dist=True)
        return iou

    def on_train_epoch_end(self):
        if (self.current_epoch+1) % self.hparams.sample_step_inc_every_n_epoch == 0:
            print("Increasing sample steps")
            self.schedule_idx = 0
            self.sample_steps = min(self.sample_steps+1, self.hparams.max_sample_steps)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )

        return optimizer
