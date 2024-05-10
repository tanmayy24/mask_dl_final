import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    U-Net model double convolution block.
    """
    def __init__(self, in_channels, out_channels):
        """Summary
        
        Args:
            in_channels (int): Description
            out_channels (int): Description
        """
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """
    U-Net model.
    """
    def __init__(self, in_channels, num_classes, filters=[64, 128, 256, 512]):
        """Summary
        
        Args:
            in_channels (int): Description
            num_classes (int):
            filters (list, optional): Description
        """
        super().__init__()

        # pooling
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # encoding layers
        self.down1 = ConvBlock(in_channels, filters[0])
        self.down2 = ConvBlock(filters[0], filters[1])
        self.down3 = ConvBlock(filters[1], filters[2])
        self.down4 = ConvBlock(filters[2], filters[3])

        # bottleneck layer
        self.bottleneck = ConvBlock(filters[3], 2 * filters[3])

        # decoding layers
        self.tconv1 = nn.ConvTranspose2d(2 * filters[3], filters[3],
                                         kernel_size=2, stride=2)
        self.up1 = ConvBlock(2 * filters[3], filters[3])
        self.tconv2 = nn.ConvTranspose2d(filters[3], filters[2],
                                         kernel_size=2, stride=2)
        self.up2 = ConvBlock(filters[3], filters[2])
        self.tconv3 = nn.ConvTranspose2d(filters[2], filters[1],
                                         kernel_size=2, stride=2)
        self.up3 = ConvBlock(filters[2], filters[1])
        self.tconv4 = nn.ConvTranspose2d(filters[1], filters[0],
                                         kernel_size=2, stride=2)
        self.up4 = ConvBlock(filters[1], filters[0])

        self.map = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        # collect skip connections
        skip = []

        # downward path
        # double convolution block
        # followed by max pooling
        x = self.down1(x)
        skip.append(x)
        x = self.pool(x)

        x = self.down2(x)
        skip.append(x)
        x = self.pool(x)

        x = self.down3(x)
        skip.append(x)
        x = self.pool(x)

        x = self.down4(x)
        skip.append(x)
        x = self.pool(x)

        # bottleneck
        x = self.bottleneck(x)

        # upward path
        # transpose convolution
        # add skip connections by concatenating on channel axis
        x = self.tconv1(x)
        x = torch.cat((skip[-1], x), dim=1)
        x = self.up1(x)

        x = self.tconv2(x)
        x = torch.cat((skip[-2], x), dim=1)
        x = self.up2(x)

        x = self.tconv3(x)
        x = torch.cat((skip[-3], x), dim=1)
        x = self.up3(x)

        x = self.tconv4(x)
        x = torch.cat((skip[-4], x), dim=1)
        x = self.up4(x)

        # collapse to ouptut segmentation map
        x = self.map(x)

        return x
