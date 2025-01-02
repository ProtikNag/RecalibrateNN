# Utility: Dataset class for TCAV
import os
from PIL import Image
from torch.utils.data import Dataset


class ConceptDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = []
        self.labels = []

        for label, concept in enumerate(os.listdir(folder)):
            concept_folder = os.path.join(folder, concept)
            if os.path.isdir(concept_folder):
                for img in os.listdir(concept_folder):
                    self.images.append(os.path.join(concept_folder, img))
                    self.labels.append(label)


        def __len__(self):
            return len(self.images)


        def __getitem__(self, idx):
            img_path = self.images[idx]
            label = self.labels[idx]
            img = Image.open(img_path).convert('RGB')
            if self.trasnform:
                img = self.transform(img)

            return img, label