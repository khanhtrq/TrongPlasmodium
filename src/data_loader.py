import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np

class AnnotationDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        self.labels = set()

        with open(annotation_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(self.root_dir, path)
                label = int(label)
                self.samples.append((full_path, label))
                self.labels.add(label)

        self.classes = [str(i) for i in sorted(self.labels)]
        self.imgs = self.samples
        self.loader = lambda path: Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self.loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label
