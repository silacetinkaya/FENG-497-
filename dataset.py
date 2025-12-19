import os
import cv2
import torch
from torch.utils.data import Dataset

class PolypDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # list all filenames in images_dir
        # Only take image files (skip hidden files and others)
        valid_exts = (".jpg", ".jpeg", ".png")
        self.ids = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(valid_exts)
        ]
        self.ids.sort()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.images_dir, img_id)
        mask_path = os.path.join(self.masks_dir, img_id)

        # Read image (BGR is returned)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize everything to fixed size (e.g., 256x256)
        target_size = (256, 256)  # (width, height)
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size)

        # If you want apply transform here (none for now)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # [H, W, C] → [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # [H, W] → [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return image, mask
