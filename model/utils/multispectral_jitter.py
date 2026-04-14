import torch

class MultispectralJitter(torch.nn.Module):
    def __init__(self, brightness=0.2, contrast=0.2):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, img):
        # Expects [B, C, H, W] or [C, H, W]
        device = img.device

        b_dims = [1] * img.ndim

        # Generate one contrast factor for all bands
        contrast_factor = torch.empty(b_dims, device=device).uniform_(1 - self.contrast, 1 + self.contrast)
        img = img * contrast_factor

        # Generate one brightness offset for all bands
        brightness_factor = torch.empty(b_dims, device=device).uniform_(-self.brightness, self.brightness)
        img = img + brightness_factor

        return img
