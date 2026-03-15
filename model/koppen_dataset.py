import torch
import numpy as np
from torch.utils.data import IterableDataset


class KoppenDataset(IterableDataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = tf_dataset

    def __iter__(self):
        for images, labels in self.tf_dataset.as_numpy_iterator():
            # Labels: Shift from 1-30 to 0-29
            labels_copy = (labels - 1).copy()

            # Images: Move Channels to front for PyTorch (B, H, W, C) -> (B, C, H, W)
            images_copy = np.transpose(images, (0, 3, 1, 2)).copy()

            yield torch.from_numpy(images_copy).float(), torch.from_numpy(labels_copy).long()
