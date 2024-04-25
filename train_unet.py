import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


class WenmaSet(Dataset):
    def __init__(self, data_path, data_type, transform=None):
        self.data_path = data_path
        self.data_type = data_type
        self.transform = transform

        if (
            self.data_type == "train"
            or self.data_type == "val"
            or self.data_type == "unlabeled"
        ):
            self.num_frames = 22

        else:
            self.num_frames = 11

    def __getitem__(self, ind):
        images = []
        masks = []

        if "train" in self.data_type:
            ind = ind

        elif "val" in self.data_type:
            ind = ind + 1000

        elif "unlabeled" in self.data_type:
            ind = ind + 2000

        elif "hidden" in self.data_type:
            ind = ind + 15000

        video_path = os.path.join(self.data_path, "video_{}".format(ind))

        if "hidden" in self.data_type or "unlabeled" in self.data_type:
            mask_path = None

        else:
            mask_path = os.path.join(video_path, "mask.npy")

        for frame in range(self.num_frames):
            image_path = os.path.join(video_path, "image_{}.png".format(frame))

            image = np.array(Image.open(image_path))

            if self.transform:
                image = self.transform(image)

            images.append(image)

            if mask_path != None:
                if "prediction" in self.data_type:
                    mask = np.load(mask_path)[frame + 11]

                else:
                    mask = np.load(mask_path)[frame]

            else:
                mask = torch.zeros((160, 240))

            masks.append(mask)

        return images, masks

    def __len__(self):
        return len(os.listdir(self.data_path))


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomErasing(value=120, inplace=True, p=1, scale=(0.0125, 0.0125)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.00001, 2)),
        transforms.Resize((160, 240)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

data_path = "/scratch/tk3309/dl_data/dataset/"

train_dataset = WenmaSet(
    data_path=data_path + "train", data_type="train", transform=transform
)

val_dataset = WenmaSet(
    data_path=data_path + "val", data_type="val", transform=transform
)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)

val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=6)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.max_pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.conv_transpose(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class WenmaNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(WenmaNet, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.decoder1 = DecoderBlock(1024, 512, 512)
        self.decoder2 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder4 = DecoderBlock(128, 64, 64)

        self.output_layer = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        b1 = self.bottleneck(p4)

        d1 = self.decoder1(b1, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        outputs = self.output_layer(d4)

        return outputs


model = WenmaNet(in_channels=3, n_classes=49).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=0.01)

scaler = torch.cuda.amp.GradScaler()

num_epochs = 100

train_losses = []
val_losses = []

cnt = 0

for epoch in tqdm(range(num_epochs)):
    model.train()

    train_loss = 0

    for images, masks in train_dataloader:
        for frame in range(22):
            image = images[frame].to(device)

            mask = masks[frame].type(torch.long).to(device)

            with torch.cuda.amp.autocast():
                mask_prediction = model(image)

                train_loss_fn = loss_fn(mask_prediction, mask)

            optimizer.zero_grad()

            scaler.scale(train_loss_fn).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += train_loss_fn.item()

            train_losses.append(train_loss)

    print("Train Loss:", train_loss)

    model.eval()

    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_dataloader:
            for frame in range(22):
                image = images[frame].to(device)

                mask = masks[frame].type(torch.long).to(device)

                with torch.cuda.amp.autocast():
                    mask_prediction = model(image)

                    val_loss_fn = loss_fn(mask_prediction, mask)

                val_loss += val_loss_fn.item()

                val_losses.append(val_loss)

    print("Val Loss:", val_loss)


epochs = range(1, len(val_losses) + 1)
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Over Epochs")
plt.legend()
plt.show()

model_scripted = torch.jit.script(model.to("cpu"))
model_scripted.save("checkpoints/unet9.pt")
