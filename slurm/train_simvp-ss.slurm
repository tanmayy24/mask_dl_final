#!/bin/bash

#SBATCH --job-name=train_simvp_ss
## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_001-2023fa-19

## change the partition number to use different number of GPUs
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

## SBATCH --partition=n1c24m128-v100-4
## SBATCH --gres=gpu:4
## SBATCH --cpus-per-task=24

#SBATCH --time=06:00:00
#SBATCH --output=logs/train_simvp_ss_%j.out
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=me2646@nyu.edu
#SBATCH --mail-type=ALL

singularity exec --nv \
	    --bind /scratch \
	    --overlay /scratch/me2646/maskpredformer.ext3:ro \
/share/apps/images/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh; python3 train_simvp_ss.py --simvp_path simvp_epoch=16-val_loss=0.014.ckpt \
--max_epochs 100 --sample_step_inc_every_n_epoch 20 --batch_size 4 --check_val_every_n_epoch 2 \
"
