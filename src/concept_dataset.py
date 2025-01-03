# Utility: Dataset class for TCAV
import os

from PIL import Image
from torch.utils.data import Dataset


class ConceptDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = []

        for img in os.listdir(self.folder):
            self.images.append(os.path.join(self.folder, img))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img
