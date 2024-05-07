### Team 3 - batch_size=3 (Deep Learning Spring'24)

# Frame Prediction for Synthetic Videos

## Overview
This project aims to predict the 22nd frame of synthetic video sequences based on the initial 11 frames. It uses deep learning models trained on videos characterized by various shapes, materials, and colors of the objects they display.

## Objective
To leverage advanced machine learning techniques to predict future video frames, improving the understanding of object behavior and motion in synthetic environments.

## Dataset
- **Type**: Synthetic videos
- **Details**: Videos featuring 48 different objects, each with unique combinations of shape, material, and color.

## Evaluation Metric
- **Intersection-over-Union (IoU)**: Also known as the Jaccard Similarity Index, it measures the accuracy of the predicted video frames against the ground truth.

## Model Architecture
1. **Segmentation**: Uses U-Net for generating accurate masks of video frames.
2. **Frame Prediction**:
   - **Initial Prediction**: SimVP model with gSTA for predicting intermediate frames.
   - **Fine-tuning**: Enhances prediction accuracy focusing on Intersection-over-Union (IoU) metric.

## Getting Started

### Prerequisites
Ensure you have Python and pip installed on your system.

### Install Dependencies
```pip install -r requirements.txt```

```
git clone git@github.com:chengtan9907/OpenSTL.git](https://github.com/chengtan9907/OpenSTL.git
```
```
cd <path_to_OpenSTL>
pip install -e .
```

### Usage

#### Training the model:
`python train.py`

This will train the initial U-Net and SimVP models and save the checkpoints.
#### Fine-tuning the model:
`python finetune.py --simvp_path <path-to-simvp-checkpoint>`

Replace <path-to-simvp-checkpoint> with the path to your trained model checkpoint from the initial training phase.

## Experimental Results

Below are the validation IoU scores obtained for each stage of the model training and fine-tuning process:

| Model                 | Validation IoU |
|-----------------------|----------------|
| U-Net                 | 0.969          |
| SimVP (Cross-entropy) | 0.369          |
| SimVP (Fine-tuned)    | 0.455          |

These results underline the effectiveness of the U-Net in segmentation tasks and illustrate the challenges and progress in frame prediction using the SimVP model.
