import torch
from torch import nn
from torchvision.models import swin_t, Swin_T_Weights

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=30, in_channels=12):
        super(SwinTransformer, self).__init__()
        self.model = swin_t(weights=Swin_T_Weights.DEFAULT)

        # Modify the Patch Embedding (Input Layer) for 12 channels
        old_patch_embed = self.model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_patch_embed.out_channels,
            kernel_size=old_patch_embed.kernel_size,
            stride=old_patch_embed.stride,
            padding=old_patch_embed.padding
        )

        # Pretrained Weight Expansion (The "Warm Start")
        with torch.no_grad():
            # Average the 3 RGB weights [96, 3, 4, 4] -> [96, 1, 4, 4]
            avg_weights = old_patch_embed.weight.mean(dim=1, keepdim=True)
            # Repeat to fill 12 channels [96, 12, 4, 4]
            new_conv.weight.copy_(avg_weights.repeat(1, in_channels, 1, 1))
            new_conv.bias.copy_(old_patch_embed.bias)

        self.model.features[0][0] = new_conv

        # Modify the Head
        self.model.head = nn.Linear(
            in_features=self.model.head.in_features,
            out_features=num_classes
        )

    def forward(self, x):
        return self.model(x)
