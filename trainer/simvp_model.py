__all__ = ['DEFAULT_MODEL_CONFIG', 'MaskSimVP']


from openstl.models.simvp_model import SimVP_Model
import torch.nn as nn
import torch

class MaskSimVP(nn.Module):
    def __init__(self, in_shape, hid_S, hid_T, N_S, N_T, model_type, pre_seq_len=11, aft_seq_len=11, drop_path=0.0, downsample=False):
        super().__init__()
        c = in_shape[1]
        self.simvp = SimVP_Model(
            in_shape=in_shape, hid_S=hid_S, 
            hid_T=hid_T, N_S=N_S, N_T=N_T, 
            model_type=model_type, drop_path=drop_path)
        self.token_embeddings = nn.Embedding(49, c)
        self.out_conv = nn.Conv2d(c, 49, 1, 1)
        self.pre_seq_len = pre_seq_len
        self.aft_seq_len = aft_seq_len
        self.downsample = downsample
        self.down_conv = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1)
        self.up_conv = nn.ConvTranspose2d(c, c, kernel_size=2, stride=2)

    def forward(self, tokens):
        x = self.token_embeddings(tokens)
        x = x.permute(0, 1, 4, 2, 3)

        if self.downsample:
            b, t, *_ = x.shape
            x = x.view(b*t, *x.shape[2:])
            x = self.down_conv(x)
            x = x.view(b, t, *x.shape[1:])

        if self.aft_seq_len == self.pre_seq_len:
            y_hat = self.simvp(x)
        elif self.aft_seq_len < self.pre_seq_len:
            y_hat = self.simvp(x)
            y_hat = y_hat[:, :self.aft_seq_len]
        elif self.aft_seq_len > self.pre_seq_len:
            d = self.aft_seq_len // self.pre_seq_len
            m = self.aft_seq_len % self.pre_seq_len
    
            y_hat = []
            cur_seq = x.clone()
            for _ in range(d):
                cur_seq = self.simvp(cur_seq)
                y_hat.append(cur_seq)
            
            if m != 0:
                cur_seq = self.simvp(cur_seq)
                y_hat.append(cur_seq[:, :m])
            
            y_hat = torch.cat(y_hat, dim=1)

        b, t, *_ = y_hat.shape
        y_hat = y_hat.view(b*t, *y_hat.shape[2:])
        if self.downsample:
            y_hat = self.up_conv(y_hat)

        y_hat_logits = self.out_conv(y_hat)

        _, _, h, w = y_hat_logits.shape
        y_hat_logits = y_hat_logits.view(b, t, 49, h, w)
        return y_hat_logits

