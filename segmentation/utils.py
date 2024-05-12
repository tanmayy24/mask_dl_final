import os
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import config


class IoUCELoss(nn.Module):
    """
    Custom loss function minimizing cross entropy loss
    while maximizing IoU. Add a smoothing constant to ensure
    that the IoU is differentiable in case both the predictions
    and ground truth are all zero (resulting in a division
    by zero -> inf).

    Adapted from:
    https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
    """
    def __init__(self, weight=None, size_average=True):
        super(IoUCELoss, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def iou(self, inputs, targets, smooth=1):
        # softargmax for multiclass classification
        inputs = F.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=config.NUM_CLASSES)

        # check dimension order for compatible arithmetic
        if targets.shape[-1] == config.NUM_CLASSES:
            targets = rearrange(targets, "b h w c -> b c h w")

        # intersection is equivalent to true positive count
        # union is the mutually inclusive area of all predictions and targets
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        # smooth to avoid ZeroDivisionError
        iou_loss = 1 - ((intersection + smooth) / (union + smooth))

        return iou_loss

    def forward(self, inputs, targets, smooth=1):
        # IoU loss
        IoU = self.iou(inputs, targets, smooth)

        # cross entropy loss
        CE = F.cross_entropy(inputs, targets, reduction='mean')

        # weighted sum
        IoU_CE = 1 * CE + 10 * IoU

        return IoU_CE


class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading each video frame and its
    associated mask for supervised learning.
    """
    def __init__(self, dir_path, normalize=True):
        """
        Args:
            dir_path (str): directory containing the video subdirectories
            normalize (bool): whether to normalize pixel values or not
        """
        super(Dataset).__init__()

        self.normalize = normalize
        self.image_paths, self.mask_paths = self.get_paths(dir_path)

    def get_paths(self, dir_path):
        """
        Args:
            dir_path (str): directory containing the video subdirectories
        
        Returns:
            list: list of paths (str) to .png image files
            list: list of tuples (str, int) with paths to
                  .npy files and mask indices
        """
        videos = [v for v in os.listdir(dir_path) if not v.startswith(".")]

        image_paths = []
        mask_paths = []
        for v in videos:
            for i in range(config.NUM_FRAMES):
                image_paths.append(os.path.join(dir_path, v, f"image_{i}.png"))
                mask_paths.append((os.path.join(dir_path, v, "mask.npy"), i))

        return image_paths, mask_paths

    def __len__(self):
        """
        Returns:
            int: number of items in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): index of item to load
        
        Returns:
            (np.ndarray): input image (height, width, channels)
            (np.ndarray): output mask (1, height, width)
        """
        img_path = self.image_paths[index]
        mask_path, mask_idx = self.mask_paths[index]

        # load image
        img = np.array(Image.open(img_path))
        if self.normalize:
            img = img / 255.  # normalize by max RGB value
        # load mask
        full_mask = np.load(mask_path)

        # check if the requested index is within the bounds of the mask array
        if mask_idx < full_mask.shape[0]:
            mask = full_mask[mask_idx, :, :]
        else:
            # handle index error - return an empty mask (no objects)
            mask = np.zeros_like(full_mask[0, :, :])

        return img, mask


class UnlabeledDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading each video for
    unsupervised learning.
    """
    def __init__(self, dir_path, normalize=True):
        """
        Args:
            dir_path (str): directory containing the video subdirectories
            normalize (bool): whether to normalize pixel values or not
        """
        super(Dataset).__init__()

        self.dir_path = dir_path
        self.normalize = normalize
        self.video_names = [v for v in os.listdir(dir_path) if not v.startswith(".")]

    def __len__(self):
        """
        Returns:
            int: number of items in the dataset
        """
        return len(self.video_names)

    def __getitem__(self, index):
        """
        Args:
            index (int): index of video to load
        
        Returns:
            (np.ndarray): video frames (config.NUM_FRAMES, height, width, channels)
            (str): name of video
        """
        video = self.video_names[index]

        frames = []
        # load image
        for i in range(config.NUM_FRAMES):
            # open frame image
            img_path = os.path.join(self.dir_path, video, f"image_{i}.png")
            img = np.array(Image.open(img_path))

            # permute to (channels, height, width)
            img = np.transpose(img, axes=(2, 0, 1))

            # add dimension for concatenating
            img = np.expand_dims(img, axis=0)

            if self.normalize:
                img = img / 255.  # normalize by max RGB value

            frames.append(img)

        frames = np.concatenate(frames, axis=0)

        return frames, video
