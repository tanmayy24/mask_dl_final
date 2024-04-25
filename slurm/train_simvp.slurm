#!/bin/bash

# SBATCH --job-name=frame_pred

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train-unet

#SBATCH --output=logs/train_simvp_%j.out
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=tk3309@nyu.edu
#SBATCH --mail-type=ALL

singularity exec --nv \
	    --bind /scratch \
	    --overlay /scratch/tk3309/DL24/overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
/bin/bash -c"
source /ext3/env.sh; python3 /scratch/tk3309/mask_dl_final/train_simvp.py \
--downsample --in_shape 11 49 160 240 --lr 1e-3 --pre_seq_len=11 --aft_seq_len=1 \
--max_epochs 20 --devices 2 --batch_size 4 \
--val_check_interval 0.5 --strategy ddp_find_unused_parameters_true
"
