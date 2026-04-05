import torch

class MultispectralJitter(torch.nn.Module):
    def __init__(self, brightness=0.2, contrast=0.2):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, img):
        # Expects [B, C, H, W] if on GPU after a batch, or [C, H, W]
        # Generate random factors on the same device as the input image
        device = img.device
        channels = img.shape[-3]  # Works for both [C,H,W] and [B,C,H,W]

        # Random Brightness (Offset) per channel
        # Use a small unsqueeze to make broadcasting work: [C, 1, 1]
        b_dims = [1] * img.ndim
        b_dims[-3] = channels

        brightness_factor = torch.empty(b_dims, device=device).uniform_(-self.brightness, self.brightness)
        img = img + brightness_factor

        # Random Contrast (Scale) per channel
        contrast_factor = torch.empty(b_dims, device=device).uniform_(1 - self.contrast, 1 + self.contrast)
        img = img * contrast_factor

        return img
