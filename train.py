from argparse import ArgumentParser
import os
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from trainer.config import DEFAULT_DATA_PATH, SEED
from trainer.trainer import MaskSimVPModule

def list_to_folder_name(l):
    """
    Join list elements with '-' to create a folder name.
    """
    return "-".join(map(str, l))


def dict_to_folder_name(d):
    """
    Convert a dictionary to a folder name by joining key-value pairs with '_', handling list values.
    """
    return "_".join(f"{k}={list_to_folder_name(v) if isinstance(v, list) else v}" for k, v in d.items())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    pl.seed_everything(SEED)

    module = MaskSimVPModule(
        in_shape=[11, 49, 160, 240],
        hid_S=64,
        hid_T=512,
        N_S=4,
        N_T=8,
        model_type="gSTA",
        data_root=args.data_root,
        batch_size=64,
        lr=1e-3,
        weight_decay=0.0,
        max_epochs=20,
        unlabeled=True,
        downsample=True,
        pre_seq_len=11,
        aft_seq_len=1
    )

    run_name = dict_to_folder_name(module.hparams.copy())
    dirpath = os.path.join("checkpoints", run_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="simvp_{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=20, accelerator="gpu", devices=4,
        strategy="ddp_find_unused_parameters_true", logger=None, fast_dev_run=False,
        log_every_n_steps=100, val_check_interval=0.5,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    ckpt_path = os.path.join(dirpath, "last.ckpt")
    trainer.fit(module, ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None)
