import os

import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.tuner import Tuner

from trainer.config import SEED, DEFAULT_DATA_PATH
from trainer.trainer_finetune import (
    MaskSimVPScheduledSamplingModule,
)

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
    parser.add_argument("--check_val_every_n_epoch", type=int, default=2)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Hyperparameters for the model
    parser.add_argument(
        "--simvp_path",
        type=str,
        required=True,
        help="checkpoint path to pretrained simvp prior",
    )
    parser.add_argument(
        "--sample_step_inc_every_n_epoch",
        type=int,
        default=20,
        help="how many epochs to increase sample step by 1",
    )
    parser.add_argument(
        "--max_sample_steps",
        type=int,
        default=5,
        help="maximum number of steps to sample from current model",
    )
    parser.add_argument(
        "--schedule_k",
        type=float,
        default=1.05,
        help="hyperparameter for inverse sigmoid schedule for sampling prob",
    )
    parser.add_argument("--schedule_type", type=str, default="exponential")
    parser.add_argument("--unlabeled", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--use_gt_data", action="store_true")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_PATH)
    # MultiGPU
    parser.add_argument("--strategy", type=str, default="auto")

    args = parser.parse_args()

    pl.seed_everything(SEED)

    mask_sim_vp_ckpt = torch.load(args.simvp_path)
    ss_params = mask_sim_vp_ckpt["hyper_parameters"]
    ss_params["data_root"]=args.data_root
    ss_params["sample_step_inc_every_n_epoch"] = args.sample_step_inc_every_n_epoch
    ss_params["max_sample_steps"] = args.max_sample_steps
    ss_params["schedule_k"] = args.schedule_k
    ss_params["unlabeled"] = args.unlabeled
    ss_params["batch_size"] = args.batch_size
    ss_params["lr"] = args.lr
    ss_params["max_epochs"] = args.max_epochs
    ss_params["use_gt_data"] = args.use_gt_data
    ss_params["schedule_type"] = args.schedule_type

    module = MaskSimVPScheduledSamplingModule(**ss_params)
    module.load_state_dict(mask_sim_vp_ckpt["state_dict"])
    print("INFO: loaded model checkpoint from MaskSimVP")

    run_name = dict_to_folder_name(
        {
            "method": "SS",
            "simvp": os.path.basename(args.simvp_path),
            "inc_every_n_epoch": ss_params["sample_step_inc_every_n_epoch"],
            "max_sample_steps": ss_params["max_sample_steps"],
            "schedule_k": ss_params["schedule_k"],
            "unlabeled": ss_params["unlabeled"],
            "use_gt_data": ss_params["use_gt_data"],
            "schedule_type": ss_params["schedule_type"],
        }
    )
    dirpath = os.path.join("checkpoints_finetune/", run_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="simvp_ss_{epoch}-{valid_last_frame_iou:.3f}",
        monitor="valid_last_frame_iou",
        save_top_k=3,
        mode="max",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=module.hparams.max_epochs,
        accelerator="gpu",
        devices=2,
        strategy=args.strategy,
        logger=None,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    ckpt_path = os.path.join(dirpath, "last.ckpt")
    trainer.fit(module, ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None)
