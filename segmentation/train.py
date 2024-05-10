import os
import numpy as np
from tqdm import tqdm
import pickle
from einops import rearrange
import torch
from torch.utils.data import DataLoader
import torchmetrics
import config
from unet import UNet
from utils import IoUCELoss, SegmentationDataset


def train(model, train_dataloader, val_dataloader, device, verbose=False):
    """
    Train U-Net model.
    
    Args:
        model (nn.Module): PyTorch U-Net model
        train_dataloader (DataLoader): data loader for training data
        val_dataloader (DataLoader): data loader for validation data
        device (torch.device): device to allocate model and data to
    """
    if verbose:
        print("Setting up training...")

    # send model to GPU if available
    model = model.to(device)

    # save best model
    best_model = None
    best_val_loss = 1e6

    # initialize training
    criterion = IoUCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # initialize softargmax
    softmax = torch.nn.Softmax(dim=1)

    # initialize metrics
    jaccard = torchmetrics.classification.JaccardIndex(task="multiclass", num_classes=config.NUM_CLASSES)

    # save history of metrics
    history = {'training': {'loss': [], 'iou': []},
               'validation': {'loss': [], 'iou': []}}

    if verbose:
        print("Starting training process...")

    epochs_no_improve = 0

    # training loop
    for epoch in range(config.NUM_EPOCHS):
        # loss values within epoch
        train_loss_epoch, val_loss_epoch = [], []

        # metric values within epoch
        train_iou_epoch, val_iou_epoch = [], []

        # enable training
        model.train()

        pbar = tqdm(train_dataloader)
        pbar.set_description("Training")

        for idx, (data, target) in enumerate(pbar):
            # data dimensions should be (batch, channel, height, width)
            if data.shape[1] != config.IN_CHANNELS:
                data = rearrange(data, "b h w c -> b c h w")

            # send data to device
            data = data.float().to(device)  # torch.float32
            target = target.type(torch.long).to(device)  # torch.int64

            # forward
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)

            iou_score = jaccard(softmax(output.to("cpu")), target.to("cpu"))

            train_loss_epoch.append(loss.item())
            train_iou_epoch.append(iou_score.item())

            pbar.set_postfix({"Loss": loss.item(), "IoU": iou_score.item()}, refresh=True)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if verbose:
            print("Computing average training metrics for this epoch...")

        # compute avg training loss and metrics for this epoch
        history['training']['loss'].append(np.mean(train_loss_epoch))
        history['training']['iou'].append(np.mean(train_iou_epoch))

        # enable evaluation
        model.eval()

        if verbose:
            print("Evaluating model on validation data...")

        with torch.no_grad():

            pbar = tqdm(val_dataloader)
            pbar.set_description("Validation")

            for idx, (data, target) in enumerate(pbar):
                # rearrange dimensions to (batch, channel, height, width)
                data = rearrange(data, "b h w c -> b c h w")

                # send data to device
                data = data.float().to(device)  # torch.float32
                target = target.type(torch.long).to(device)  # torch.int64

                # forward
                preds = model(data)
                loss = criterion(preds, target)

                iou_score = jaccard(softmax(preds.to("cpu")), target.to("cpu"))

                val_loss_epoch.append(loss.item())
                val_iou_epoch.append(iou_score.item())

        if verbose:
            print("Computing average validation metrics for this epoch...")

        # compute avg validation loss and metrics for this epoch
        history['validation']['loss'].append(np.mean(val_loss_epoch))
        history['validation']['iou'].append(np.mean(val_iou_epoch))

        # print summary for epoch
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}: Training Loss: {history['training']['loss'][-1]:.4f}, ",
              f"Training IoU: {history['training']['iou'][-1]:.4f}, ",
              f"Validation Loss: {history['validation']['loss'][-1]:.4f}, ",
              f"Validation IoU: {history['validation']['iou'][-1]:.4f}\n")

        # check if the model improved on validation dataset
        if history['validation']['loss'][-1] < best_val_loss:
            epochs_no_improve = 0
            if verbose:
                print("Model improved! Saving new model...")
            best_model = model
            torch.save(best_model.state_dict(), f'checkpoints/{config.MODEL_NAME}')
            best_val_loss = history['validation']['loss'][-1]
        else:
            epochs_no_improve += 1

        # save history
        with open('history.pkl', 'wb') as f:
            pickle.dump(history, f)

        # implement early stopping
        if (epochs_no_improve > config.MAX_PATIENCE) and config.EARLY_STOP:
            print(f"Model has not improved for the last {epochs_no_improve} epochs.")
            print("Stopping training early!")
            break

    return best_model, history


if __name__ == '__main__':
    model = UNet(config.IN_CHANNELS, config.NUM_CLASSES)

    train_dataset = SegmentationDataset(config.TRAIN_DATA_DIR)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                  num_workers=config.NUM_WORKERS, shuffle=True)

    val_dataset = SegmentationDataset(config.VAL_DATA_DIR)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                                num_workers=config.NUM_WORKERS, shuffle=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {DEVICE} <<<")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model, history = train(model, train_dataloader, val_dataloader, DEVICE, verbose=True)
    print("Training completed successfully.")
