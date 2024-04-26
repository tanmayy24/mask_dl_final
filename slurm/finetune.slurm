#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=finetune_fp

#SBATCH --output=logs/finetune_%j.out
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=tk3309@nyu.edu
#SBATCH --mail-type=ALL

singularity exec --nv \
	    --overlay /scratch/tk3309/DL24/overlay-50G-10M.ext3:rw \
	    /scratch/tk3309/DL24/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; python /scratch/tk3309/mask_dl_final/train_simvp_finetune.py --simvp_path /scratch/tk3309/mask_dl_final/slurm/checkpoints/in_shape=11-49-160-240_hid_S=64_hid_T=512_N_S=4_N_T=8_model_type=gSTA_batch_size=4_lr=0.001_weight_decay=0.0_max_epochs=20_pre_seq_len=11_aft_seq_len=1_unlabeled=False_downsample=True/simvp_epoch=19-val_loss=0.017-v1.ckpt \
		--max_epochs 100  --devices 2 --use_gt_data=True --sample_step_inc_every_n_epoch 20 --batch_size 4 --check_val_every_n_epoch 2"
