import torch
from torch import nn
from torchvision.models import SwinTransformer


class SwinClimateTransformer(nn.Module):
    def __init__(self, num_classes=30, in_channels=12):
        super(SwinClimateTransformer, self).__init__()
        self.model = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 2, 2], # Reduce layer 3 depth to decrease overfitting
            num_heads=[3, 6, 12, 24],
            window_size=[8, 8],
            dropout=0.2,
            stochastic_depth_prob=0.1,
            num_classes=num_classes
        )

        # Modify the Patch Embedding for 12 channels
        old_patch_embed = self.model.features[0][0]

        self.model.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_patch_embed.out_channels,
            kernel_size=old_patch_embed.kernel_size,
            stride=old_patch_embed.stride,
            padding=old_patch_embed.padding
        )

        # Pretrained Weight Expansion (Averaging + Repeating)
        with torch.no_grad():
            avg_weights = old_patch_embed.weight.mean(dim=1, keepdim=True)
            self.model.features[0][0].weight.copy_(avg_weights.repeat(1, in_channels, 1, 1))
            self.model.features[0][0].bias.copy_(old_patch_embed.bias)

        # Modify the Head
        self.model.head = nn.Linear(
            in_features=self.model.head.in_features,
            out_features=num_classes
        )

    def forward(self, x):
        return self.model(x)
