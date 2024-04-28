__all__ = ['MaskSimVPModule']
import torch
import os

import lightning as pl

import matplotlib.pyplot as plt
import wandb
import random

from .simvp_model import MaskSimVP
from .loader import DLDataset

class MaskSimVPModule(pl.LightningModule):
    def __init__(self, 
                 in_shape, hid_S, hid_T, N_S, N_T, model_type,
                 batch_size, lr, weight_decay, max_epochs,
                 data_root, pre_seq_len=11, aft_seq_len=11,
                 drop_path=0.0, unlabeled=False, downsample=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = MaskSimVP(
            in_shape, hid_S, hid_T, N_S, N_T, model_type, downsample=downsample, drop_path=drop_path,
            pre_seq_len=pre_seq_len, aft_seq_len=aft_seq_len
        )
        self.train_set = DLDataset(data_root, "train", unlabeled=unlabeled, use_gt_data=True, pre_seq_len=pre_seq_len, aft_seq_len=aft_seq_len)
        self.val_set = DLDataset(data_root, "val", use_gt_data=True, pre_seq_len=pre_seq_len, aft_seq_len=aft_seq_len)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, 
            num_workers=1, shuffle=True, pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set, batch_size=self.hparams.batch_size, 
            num_workers=1, shuffle=False, pin_memory=True
        )

    @torch.no_grad()
    def sample_autoregressive(self, x, t):
        cur_seq = x.clone()
        for _ in range(t):
            y_hat_logit_t = self.model(cur_seq)
            y_hat = torch.argmax(y_hat_logit_t, dim=2) # get current prediction
            cur_seq = torch.cat([cur_seq[:, 1:], y_hat], dim=1) # autoregressive concatenation
        return cur_seq

    def step(self, x, y):
        y_hat_logits = self.model(x)
        return y_hat_logits
    
    def training_step(self, batch, batch_idx): 
        x, y = batch
        y_hat_logits = self.step(x, y)
        
        # Flatten batch and time dimensions
        b, t, *_ = y_hat_logits.shape
        y_hat_logits = y_hat_logits.view(b*t, *y_hat_logits.shape[2:])
        y = y.view(b*t, *y.shape[2:])

        loss = self.criterion(y_hat_logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self.step(x, y)

        # Flatten batch and time dimensions
        b, t, *_ = y_hat_logits.shape
        y_hat_logits = y_hat_logits.view(b*t, *y_hat_logits.shape[2:])
        y = y.view(b*t, *y.shape[2:])
       
        loss = self.criterion(y_hat_logits, y)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr,
            total_steps=self.hparams.max_epochs*len(self.train_dataloader()),
            final_div_factor=1e4
        )
        opt_dict = {
            "optimizer": optimizer,
            "lr_scheduler":{
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            } 
        }

        return opt_dict