# Team 19 - We Need More Than Attention (WENMA)

Placed 2nd out of 38 teams in NYU Deep Learning final project in Fall 2023.

## Instructions

### Setup

1. Installing pip packages

The following packages are required to run the code. Training and inference is done on `Ubuntu 22.04` machines using `Python 3.10.12`. This setup should work on any Linux machine with recent versions of Python.

```
torch==2.1.1
torchvision==0.16.1
pytorch-lightning==2.1.2
numpy==1.26.1
tqdm
matplotlib==3.8.2
wandb==0.16.0
imageio==2.33.0
```
2. Installing OpenSTL

We use OpenSTL for the implementation of SimVP. To install OpenSTL, run the following commands:

- In a suitable directory, clone the OpenSTL repository:
```
git clone git@github.com:chengtan9907/OpenSTL.git
```
- Install OpenSTL:
```
cd <path_to_OpenSTL>
pip install -e .
```

3. Wandb setup

We use wandb for logging. To setup wandb, run the following commands:

```
wandb login
```

Now, you can start training the models.

### Training and Inference

0. Place the data (or symlink) it to `data/Dataset_Student`. This directory should contain `train`, `val` and `unlabeled` folders.

1. The first step is to train a UNet with these data: `python3 train_unet.py`. This will save the model in `checkpoints/unet9.pt`.

2. Now, we can generate masks from the data.

- To generate masks for train and val splits, run 
`python generate_masks.py --model_checkpoint checkpoints/unet9.pt --data_root data/Dataset_Student --split <train, val> --output_file <data/DL/train_masks.pt, data/DL/val_masks.pt>`
- For training the world model on the labeled data only, you can run for `train` and `val` splits.

- For this step, we also need to merge all ground truth masks into one file. To do this, run 
`python merge_masks.py --data_root data/Dataset_Student --split <train, val> --output_file <data/DL/train_gt_masks.pt, data/DL/val_gt_masks.pt>` for `train` and `val` splits.

Or, you can get the pre-generated masks from here (this link requires an NYU account):
| Split | Link |
| ------------- | ------------- |
| Train  | [Link](https://drive.google.com/file/d/1T3tFfziIjQhSiwSEaJJQSx11x6MOJmla/view?usp=sharing)  |
| Validation  | [Link](https://drive.google.com/file/d/1FGxuEG-IZdVe3dDPE1AKj0BYn4ys3g_t/view?usp=sharing) |

3. Now, we can train our prediction model on the generated masks.

- For training only on labeled set:
`python3 train_simvp.py --downsample --in_shape 11 49 160 240 --lr 1e-3 --pre_seq_len=11 --aft_seq_len=1 --max_epochs 20 --batch_size 4 --check_val_every_n_epoch=1`

- To train on labeled and unlabeled set, generate masks for unlabeled and add `--unlabeled` flag.

4. Now, we can finetune with scheduled sampling.

`python3 train_simvp_ss.py --simvp_path checkpoints/simvp_epoch=16-val_loss=0.014.ckpt --sample_step_inc_every_n_epoch 20 --max_epochs 100 --batch_size 4 --check_val_every_n_epoch 2`

We used the checkpoint after second epoch of scheduled sampling for our final submission. The checkpoints are here:

**Checkpoints**
| Name | Link |
| ------------- | ------------- |
| Best w/o scheduled sampling  | [Link](https://drive.google.com/file/d/1RpfAS9w553nD3H6gdQKEvNRoK0IiqSKz/view?usp=sharing)  |
| Best after  | [Link](https://drive.google.com/file/d/1Gqd8eBK-0JRXhStSZr__vGgfhY_3KXDb/view?usp=sharing)  |


5. To generate predictions on the hidden set, run this notebook: `nbs/96_get_results_combined_with_unet.ipynb`

The final predictions are [here](https://drive.google.com/file/d/1uUaAZlHKhOSbLRuy9wwdOfQEIqp8zrkL/view?usp=sharing).
