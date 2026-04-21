import torch
import torch.nn as nn
from torchvision import models

class ClimateCNN(nn.Module):
    def __init__(self, num_classes=30, in_channels=12):
        super(ClimateCNN, self).__init__()

        # Using pre-defined ResNet50 CNN architecture
        self.model = models.resnet50(weights='DEFAULT')

        # Modify first layer for 12 spectral bands input depth
        with torch.no_grad():
            old_weight = self.model.conv1.weight.data  # Shape: [64, 3, 7, 7]
            # Create new weight tensor for 12 channels
            new_weight = torch.cat([old_weight] * 4, dim=1)  # Shape: [64, 12, 7, 7]
            # Scale by 3/12 to keep the mean activation stable
            new_weight = new_weight * (3.0 / 12.0)

        self.model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.model.conv1.weight.data = new_weight

        # Modify full-connected layer for number of output classes and add dropout
        num_features = int(self.model.fc.in_features)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class ClimateCNNk3(ClimateCNN):
    def __init__(self, num_classes=30, in_channels=12):
        super(ClimateCNNk3, self).__init__(num_classes=num_classes, in_channels=in_channels)

        with torch.no_grad():
            # ClimateCNN uses [64, 12, 7, 7] weights
            parent_weights = self.model.conv1.weight.data
            # Take the center 3x3 slice of the 7x7 weights to seed new layer
            seed_weight = parent_weights[:, :, 2:5, 2:5]

        # Modify first conv layer (same receptive field, more conv layers)
        self.model.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False # No bias before BatchNorm
            )
        )

        # Apply the seed weights to the first layer of the sequence
        with torch.no_grad():
            self.model.conv1[0].weight.copy_(seed_weight)
