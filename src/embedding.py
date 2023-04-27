"""
Embedding model for 64x64x4 latents of SD2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import autograd.numpy as np

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1(x)


class DoubleConv(nn.Module):
    """
    Layer that applies (convolution, batchnorm, relu) operations twice
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of channels of input data
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward method of DoubleConv
        """
        return self.double_conv(x)

class DoubleConv3d(nn.Module):
    """
    Layer that applies (convolution, batchnorm, relu) operations twice on 3d data
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of channels of input data
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same", bias=False),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(out_channels, out_channels, kernel_size=3, padding="same", bias=False),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward method of DoubleConv
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    Layer that applies downscaling by using maxpool then double conv operations
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of channels of input data
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
        Forward method of Down
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Layer that applies upscaling by using transpose2d then double conv operations
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of channels of input data
            out_channels (int): Number of output channels
        """
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward method of Up
        Firstly x1 is upsampled and padded to the same dimensions as x2, then x1 and x2 are concatenated and DoubleConv is applied
        This is used to apply skip connections 
        Args:
            x1 (np.ndarray): Image array gotten from previous Unet layer
            x2 (np.ndarray): Image array gotten from between downsampling layers
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpSingle(nn.Module):
    """
    Layer that applies upscaling by using transpose2d then double conv operations
    WITHOUT skip connection
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of channels of input data
            out_channels (int): Number of output channels
        """
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        """
        Forward method of UpSingle
        """
        out = self.up(x)
        return self.conv(out)

class OutConv(nn.Module):
    """
    Conv2d layer for output
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of channels of input data
            out_channels (int): Number of output channels
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward method of OutConv
        """
        return self.conv(x)

class CNN(nn.Module):
    """
    Basic CNN for regression
    """
    def __init__(self, dims):
        super().__init__()
        self.inc = DoubleConv(3, 128)
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        self.down3 = Down(512,1024)
        self.down4 = Down(1024,2048)

        in_features = int(2048*dims**2/256)

        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,64)
        self.fc4 = nn.Linear(64,1)

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.fc1 = nn.Linear(32*32*32, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 1)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return self.act(x)
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.act(self.fc4(x))
        return x.squeeze(1)