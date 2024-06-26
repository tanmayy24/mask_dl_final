### Deep Learning Spring 2024
### Team 3: `batch_size=3`

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
- **Intersection-over-Union (IoU)**: Also known as the Jaccard Index, it measures how well the predicted mask of the video frame matches with the ground truth. We use the `JaccardIndex` module from `torchmetrics` ([Lightning AI](https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html)) to this metric.

## Model Architecture
1. **Segmentation**: Uses the U-Net model [[1]](#1) for generating accurate masks of all unlabeled video frames.
2. **Frame Prediction**:
   - **Initial Prediction**: SimVP [[2]](#2) model with gSTA for predicting future masks.
   - **Fine-tuning**: Enhances prediction accuracy by focusing on the IoU metric.

## Getting Started

### Prerequisites
The set-up requires Python 3 and pip.

### Install Dependencies
Install required libraries.
```
pip install -r requirements.txt

```

Install [OpenSTL](https://github.com/chengtan9907/OpenSTL) to use SimVP.
```
git clone https://github.com/chengtan9907/OpenSTL.git
cd <path/to/OpenSTL>

pip install -e .
```

## Usage

### Segmentation

#### Configuration

First, configure the model training and inference.

```
cd segmentation/
```

Open `config.py` and change the values of the `TRAIN_DATA_DIR`, `VAL_DATA_DIR`, and `UNLABELED_DATA_DIR` constants to the locations of your `train/`, `val/`, and `unlabeled/` directories. Use absolute paths to avoid errors. Set `LABELED_OUT_DIR` to the location where you would like the predicted masks of the unlabeled data saved.

#### Train the U-Net Model

Run the training script.

```
python train.py
```

This will save the best model to the `checkpoints/` directory with the `MODEL_NAME` specified in `config.py`.

#### Generate Masks

Generate masks for unlabeled data.

```
python label.py
```

### Frame Prediction

#### Configuration

From the root directory, navigate to `trainer/config.py` and set `DEFAULT_DATA_PATH` to the directory of where all of the data is stored.


#### Train the Model:

```
python train.py
```

This will train the initial SimVP model directly on the masks and save the checkpoints.

#### Fine-tune the Model:

```
python finetune.py --simvp_path <path/to/simvp/checkpoint>
```

#### Generate Predictions

To generate predictions on the validation set, update the `"ckpt_path"` in `predict_simvp_finetune_val.py` to the path of the best trained model.

Then, run the following to print the IoU and save the predictions.
```
python predict_simvp_finetune_val.py
```

Similarly, to generate predictions on the hidden set, update the `"ckpt_path"` in `predict_simvp_finetune_hidden.py` to the path of the best trained model.

Then, run the following to save predictions on the hidden set.
```
python predict_simvp_finetune_hidden.py
```

## Experimental Results

Below are the validation IoU scores obtained for each stage of the model training and fine-tuning process:

| Model                 | Validation IoU |
|-----------------------|----------------|
| U-Net                 | 0.969          |
| SimVP (Cross-entropy) | 0.369          |
| SimVP (Fine-tuned)    | 0.455          |

These results underline the effectiveness of the U-Net in segmentation tasks and illustrate the challenges and progress in frame prediction using the SimVP model.

## References

<a id="1">[1]</a> 
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image segmentation. In _Medical Image Computing and Computer-assisted Intervention-MICCAI 2015: 18th International Conference_, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.

<a id="2">[2]</a> 
Gao, Z., Tan, C., Wu, L., & Li, S. Z. (2022). SimVP: Simpler Yet Better Video Prediction. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ (pp. 3170-3180).

Our work was inspired by the workflow proposed by the **[maskpredformer](https://github.com/eneserciyes/maskpredformer)**.
