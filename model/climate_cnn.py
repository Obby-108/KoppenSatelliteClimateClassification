import torch.nn as nn
from torchvision import models

class ClimateCNN(nn.Module):
    def __init__(self, num_classes=30):
        super(ClimateCNN, self).__init__()

        # Using pre-defined ResNet18 CNN architecture
        self.model = models.resnet18(weights=None)

        # Modify first layer for 12 spectral bands input depth
        self.model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify full-connected layer for number of output classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
