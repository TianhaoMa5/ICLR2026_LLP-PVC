import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
# LeNet-5 网络结构定义
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 输入 1x28x28，输出 6x28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)               # 输出 6x14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)           # 输出 16x10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)               # 输出 16x5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)                            # 全连接层，输入 400，输出 120
        self.fc2 = nn.Linear(120, 84)                                    # 全连接层，输入 120，输出 84
        self.fc3 = nn.Linear(84, num_classes)                            # 全连接层，输入 84，输出 10

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5Plus(nn.Module):
    def __init__(self, num_classes=10, c1=32, c2=64, fc1=256, fc2=128, drop=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 5, stride=1, padding=2)
        self.bn1   = nn.BatchNorm2d(c1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(c1, c2, 5, stride=1)   # 28->14->10
        self.bn2   = nn.BatchNorm2d(c2)
        self.pool2 = nn.MaxPool2d(2, 2)               # 10->5

        self.fc1 = nn.Linear(c2 * 5 * 5, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, num_classes)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)

class MLPDropIn(nn.Module):
    def __init__(self, num_classes=10, use_dropout=False, p=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(p) if use_dropout else nn.Identity()
        self.fc1 = nn.Linear(28*28, num_classes)  # 784 -> num_classes
        self.fc2 = nn.Identity()
        self.fc3 = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc1(x)   # logits
        x = self.fc2(x)   # 占位，不做事
        x = self.fc3(x)   # 占位，不做事
        return x
class LeNet(nn.Module):
    """
    LeNet: A simple convolutional neural network for image classification.

    This implementation is adapted for 32x32 RGB images, such as CIFAR-10.
    """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)  # RGB input (3 channels)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # Output: 16 channels

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)  # Input: Flattened feature maps
        self.fc2 = nn.Linear(in_features=120, out_features=84)          # Hidden layer
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes) # Output layer (num_classes)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input tensor with shape (batch_size, 3, 32, 32).

        Returns:
            Tensor: Output tensor with shape (batch_size, num_classes).
        """
        # Layer 1: Convolution -> ReLU -> Max Pool
        x = F.relu(self.conv1(x))  # Shape: (batch_size, 6, 28, 28)
        x = self.pool(x)           # Shape: (batch_size, 6, 14, 14)

        # Layer 2: Convolution -> ReLU -> Max Pool
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 16, 10, 10)
        x = self.pool(x)           # Shape: (batch_size, 16, 5, 5)

        # Flatten the feature maps for the fully connected layers
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 16*5*5)

        # Fully connected layers
        x = F.relu(self.fc1(x))    # Shape: (batch_size, 120)
        x = F.relu(self.fc2(x))    # Shape: (batch_size, 84)
        x = self.fc3(x)            # Shape: (batch_size, num_classes)

        return x