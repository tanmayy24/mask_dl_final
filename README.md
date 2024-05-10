### Team 3: `batch_size=3` (Deep Learning Spring 2024)

# Frame Prediction and Segmentation for Synthetic Videos

## Overview
This project aims to predict and segment the 22nd frame of synthetic video sequences based on the initial 11 frames. It uses deep learning models trained on videos of objects with various shapes, materials, and colors.

## Objective
To leverage advanced machine learning techniques to predict future video frames and produce semantically segmented masks for the final frame. We hope to improve the understanding of object behavior and motion in synthetic environments.

## Dataset
- **Data**: RGB frames of videos with simple 3D shapes interacting with each other while following the simple principles of physics (`.png`).
- **Labels**: Semantic segmentation mask for each frame (`.npy`).
- **Quantity**:
    - 1,000 training videos with 22 frames each and their corresponding labeled masks.
    - 1,000 validation videos with 22 frames each and their corresponding labeled masks.
    - 13,000 unlabeled videos with 22 frames each.
    - 5,000 (hidden) unlabeled videos with 11 frames each (for evaluation).
- **Details**: The videos feature 48 different objects, each with a unique combination of shape, material, and color.
- **Structure**:

  ```
  data/
    ├── train/
        ├── video_00000/
            ├── image_0.png
                ...
            ├── image_21.png
            ├── mask.npy
            ...
        ├── video_00999/
    ├── val/
        ├── video_01000/
            ├── image_0.png
                ...
            ├── image_21.png
            ├── mask.npy
            ...
        ├── video_01999/
    ├── unlabeled/
        ├── video_02000/
            ├── image_0.png
                ...
            ├── image_21.png
            ...
        ├── video_14999/
    ├── hidden/
            ├── video_15000/
                ├── image_0.png
                    ...
                ├── image_10.png
                ...
            ├── video_19999/
  ```

## Evaluation Metric
- **Intersection-over-Union (IoU)**: Also known as the Jaccard Index, it measures how well the predicted mask of the video frame matches with the ground truth. We use the `JaccardIndex` module from `torchmetrics` ([Lightning AI](https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html)).

## Model Architecture
1. **Segmentation**: Uses the U-Net model for generating accurate masks of video frames.
2. **Frame Prediction**:
   - **Initial Prediction**: SimVP model with gSTA for predicting intermediate frames.
   - **Fine-tuning**: Enhances prediction accuracy focusing on Intersection-over-Union (IoU) metric.

## Getting Started

### Prerequisites
Ensure you have Python and pip installed on your system.

### Install Dependencies
```pip install -r requirements.txt```

```
git clone https://github.com/chengtan9907/OpenSTL.git
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
