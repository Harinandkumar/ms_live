import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class MSDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = os.listdir(image_dir)

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(
            self.image_dir,
            self.images[index]
        )

        mask_path = os.path.join(
            self.mask_dir,
            self.images[index]
        )

        image = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = image / 255.0
        mask = mask / 255.0

        image = np.expand_dims(
            image,
            axis=0
        )

        mask = np.expand_dims(
            mask,
            axis=0
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )

        return image, mask