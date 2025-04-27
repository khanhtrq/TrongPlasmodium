import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np

class AnnotationDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None, class_names=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        self.labels = set()
        self.targets = [] # Add list to store targets

        with open(annotation_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(self.root_dir, path)
                label = int(label)
                self.samples.append((full_path, label))
                self.labels.add(label)
                self.targets.append(label) # Store the label

        # Debug print to show unique labels
        print(f"🧩 Unique labels in annotation file: {sorted(self.labels)}")

        # Use provided class_names if available, otherwise generate numeric class names
        if class_names is not None:
            self.classes = class_names
            # Remap label indices to contiguous zero-based indices matching class_names
            # Assume class_names are in the intended order
            unique_labels = sorted(self.labels)
            if unique_labels != list(range(len(class_names))):
                print(f"⚠️ Remapping label indices to match class_names order.")
                label_to_index = {orig: idx for idx, orig in enumerate(unique_labels)}
                self.samples = [(img_path, label_to_index[label]) for img_path, label in self.samples]
                self.targets = [label_to_index[label] for label in self.targets]
                print(f"🔁 Label mapping: {label_to_index}")
        else:
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
