import os
import numpy as np
from einops import rearrange
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import config
from unet import UNet
from utils import UnlabeledDataset


def label(model, dataloader, out_dir, device):
    """
    Run U-Net model on video frames to generate segmentation masks.
    This function loads all of the frames of a single video and
    produces a mask for each frame. These masks are saved as a single
    .npy file: video_#####_mask.npy
    
    Args:
        model (nn.Module): trained PyTorch U-Net model
        train_dataloader (DataLoader): data loader for unlabeled data
        out_dir (str): location to save generated masks
        device (torch.device): device to allocate model and data to
    """
    assert dataloader.batch_size == 1, "BATCH_SIZE for inference must be 1"

    # send model to GPU if available
    model = model.to(device)

    # initialize softargmax
    softmax = torch.nn.Softmax(dim=1)

    # enable evaluation
    model.eval()

    with torch.no_grad():

        pbar = tqdm(dataloader)

        for idx, (data, video_name) in enumerate(pbar):
            # should be using a batch size of 1
            # squeeze first dimension
            if data.shape[0] == 1:
                data = torch.squeeze(data, dim=0)

            # data dimensions should be (frames, channel, height, width)
            if data.shape[1] != config.IN_CHANNELS:
                data = rearrange(data, "b h w c -> b c h w")

            # send data to device
            data = data.float().to(device)  # torch.float32

            # forward
            preds = model(data)

            # apply softmax
            mask = torch.argmax(softmax(preds), dim=1)

            # save mask
            save_path = os.path.join(out_dir, f"{video_name[0]}_mask.npy")
            with open(save_path, 'wb') as f:
                np.save(f, mask.cpu())


if __name__ == '__main__':
    # set up data
    # MUST USE A BATCH SIZE OF 1 FOR DATALOADER TO WORK CORRECTLY
    print("Setting up dataloader...")
    dataset = UnlabeledDataset(config.UNLABELED_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=1,
                            num_workers=config.NUM_WORKERS, shuffle=False)

    # get device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {DEVICE} <<<")

    # load trained model
    print("Loading model...")
    model = UNet(config.IN_CHANNELS, config.NUM_CLASSES)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(f'checkpoints/{config.MODEL_NAME}'))
    print("Model loaded!")

    # run inference
    os.makedirs(config.LABELED_OUT_DIR, exist_ok=True)
    mask_list = label(model, dataloader, config.LABELED_OUT_DIR, DEVICE)
