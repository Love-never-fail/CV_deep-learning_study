import torch
import torch.nn as nn

# AlexNet 구현

class AlexNet(nn.Module):
  def __init__(self, num_classes=1000):
    super(AlexNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=5)
    self.conv2 = nn.Conv2d(64, 192, 5, padding=2)
    self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
    self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
    self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.flatten = nn.Flatten()
    self.l1 = nn.Linear(256, num_classes)

    self.features = nn.Sequential(
        self.conv1,
        self.relu,
        self.maxpool,
        self.conv2,
        self.relu,
        self.maxpool,
        self.conv3,
        self.relu,
        self.conv4,
        self.relu,
        self.conv5,
        self.relu,
        self.maxpool,
    )

  def forward(self, x):
    x1 = self.features(x)
    x2 = self.flatten(x1)
    x3 = self.l1(x2)
    return x3  
