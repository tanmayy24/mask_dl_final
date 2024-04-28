#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=frame_pred

#SBATCH --output=logs/train_simvp_%j.out
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=tk3309@nyu.edu
#SBATCH --mail-type=ALL

singularity exec --nv \
	    --overlay /scratch/tk3309/DL24/overlay-50G-10M.ext3:rw \
	    /scratch/tk3309/DL24/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /scratch/tk3309/mask_dl_final/train_simvp.py --unlabeled \
    	--downsample --lr 1e-3 --pre_seq_len=11 --aft_seq_len=1 \
    	--max_epochs 20 --devices 2 --batch_size 64 \
   		--val_check_interval 0.5 --strategy ddp_find_unused_parameters_true"
