import os

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from trainer.config import DEFAULT_DATA_PATH, SEED
from trainer.trainer import MaskSimVPModule

def list_to_folder_name(l):
    return "-".join([str(x) for x in l])


def dict_to_folder_name(d):
    return "_".join(
        [
            f"{k}={list_to_folder_name(v) if isinstance(v, list) else v}"
            for k, v in d.items()
        ]
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # Trainer arguments
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--fast_dev_run", action="store_true")

    # Hyperparameters for the model
    parser.add_argument("--unlabeled", action="store_true")
    parser.add_argument("--downsample", action="store_true")
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hid_S", type=int, default=64)
    parser.add_argument("--hid_T", type=int, default=512)
    parser.add_argument("--N_S", type=int, default=4)
    parser.add_argument("--N_T", type=int, default=8)
    parser.add_argument("--model_type", type=str, default="gSTA")
    parser.add_argument("--in_shape", type=int, default=[11, 49, 160, 240], nargs="+")
    parser.add_argument("--pre_seq_len", type=int, default=11)
    parser.add_argument("--aft_seq_len", type=int, default=11)

    # MultiGPU
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default='auto')

    args = parser.parse_args()

    pl.seed_everything(SEED)
    module = MaskSimVPModule(
        in_shape=args.in_shape,
        hid_S=args.hid_S,
        hid_T=args.hid_T,
        N_S=args.N_S,
        N_T=args.N_T,
        model_type=args.model_type,
        data_root=args.data_root,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        unlabeled=args.unlabeled,
        downsample=args.downsample,
        drop_path=args.drop_path,
        pre_seq_len=args.pre_seq_len,
        aft_seq_len=args.aft_seq_len
    )
    hparams = module.hparams.copy()
    del hparams["data_root"]
    del hparams["drop_path"]
    run_name = dict_to_folder_name(hparams)
    dirpath = os.path.join("checkpoints/", run_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="simvp_{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=module.hparams.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        strategy=args.strategy,
        logger=None,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    ckpt_path = os.path.join(dirpath, "last.ckpt")
    trainer.fit(module, ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None)
