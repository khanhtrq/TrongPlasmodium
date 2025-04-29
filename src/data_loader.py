import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import warnings

class AnnotationDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None, class_names=None):
        """
        Args:
            annotation_file (str): Path to the annotation file (e.g., 'train_annotation.txt').
                                   Each line: relative/path/to/image.jpg label_index
            root_dir (str): Root directory where images are located.
            transform (callable, optional): Optional transform to be applied on a sample.
            class_names (list, optional): List of class names in the desired order.
                                          If provided, labels in the annotation file will be
                                          mapped to indices corresponding to this list.
        """
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        self.original_labels = set() # Store labels as read from the file
        self.targets = [] # Final (potentially remapped) label indices

        print(f"🔍 Loading annotations from: {annotation_file}")
        print(f"   Image root directory: {root_dir}")

        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Annotation file not found: {annotation_file}")

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 2:
                warnings.warn(f"⚠️ Skipping malformed line {i+1} in {annotation_file}: '{line.strip()}'")
                continue
            path, label_str = parts
            try:
                label = int(label_str)
            except ValueError:
                warnings.warn(f"⚠️ Skipping line {i+1} with non-integer label in {annotation_file}: '{line.strip()}'")
                continue

            full_path = os.path.join(self.root_dir, path)
            # Optional: Check if file exists here, but can slow down initialization
            # if not os.path.exists(full_path):
            #     warnings.warn(f"Image path not found: {full_path} (from line {i+1})")
            #     continue

            self.samples.append((full_path, label))
            self.original_labels.add(label)

        if not self.samples:
             raise ValueError(f"❌ No valid samples loaded from {annotation_file}. Check file format and paths.")

        print(f"   Found {len(self.samples)} samples.")
        print(f"   Original labels found in file: {sorted(list(self.original_labels))}")

        # --- Class Name and Label Mapping Logic ---
        unique_original_labels = sorted(list(self.original_labels))

        if class_names is not None:
            self.classes = class_names
            print(f"   Using provided class names: {self.classes}")
            num_expected_classes = len(self.classes)

            # Create mapping from original label in file to the index in class_names
            # We assume the *order* in class_names defines the final target indices (0, 1, 2...)
            # This requires that the labels in the annotation file correspond *semantically*
            # to the class names provided, even if the numeric values differ.
            # Example: file has labels [10, 20, 30], class_names is ['cat', 'dog', 'bird']
            # We need a way to know 10 means 'cat', 20 means 'dog', etc.
            # A common scenario is the file labels ARE the desired 0-based indices.

            # Let's assume the unique sorted labels from the file should map 1:1 to class_names
            if len(unique_original_labels) != num_expected_classes:
                 warnings.warn(f"⚠️ Mismatch! Found {len(unique_original_labels)} unique labels in file, but {num_expected_classes} class names provided. Label mapping might be incorrect.")
                 # Attempt a direct mapping anyway, hoping the file labels are 0..N-1
                 label_to_final_index = {label: label for label in unique_original_labels}

            else:
                 # Assume the sorted unique labels correspond to the order of class_names
                 label_to_final_index = {orig_label: idx for idx, orig_label in enumerate(unique_original_labels)}
                 print(f"   Mapping original labels to class_names indices: {label_to_final_index}")


            # Apply the mapping
            try:
                remapped_samples = []
                self.targets = []
                for img_path, orig_label in self.samples:
                    if orig_label not in label_to_final_index:
                         warnings.warn(f"⚠️ Original label {orig_label} (from {img_path}) not found in mapping keys {list(label_to_final_index.keys())}. Skipping sample.")
                         continue
                    final_label = label_to_final_index[orig_label]
                    if not (0 <= final_label < num_expected_classes):
                         warnings.warn(f"⚠️ Mapped label {final_label} (from original {orig_label}) is outside the expected range [0, {num_expected_classes-1}] for class_names. Skipping sample.")
                         continue
                    remapped_samples.append((img_path, final_label))
                    self.targets.append(final_label)
                self.samples = remapped_samples
                if not self.samples:
                    raise ValueError("❌ No samples remained after label remapping. Check consistency between annotation file labels and provided class_names.")
                print(f"   {len(self.samples)} samples remain after remapping.")

            except KeyError as e:
                 raise ValueError(f"❌ Error remapping labels. Original label {e} from annotation file not found in the derived mapping. Ensure consistency.")

        else:
            # Infer class names from sorted unique labels
            self.classes = [str(i) for i in unique_original_labels]
            print(f"   Inferring class names from labels: {self.classes}")
            num_expected_classes = len(self.classes)
            # Create a mapping from original label to a 0-based contiguous index
            label_to_final_index = {orig_label: idx for idx, orig_label in enumerate(unique_original_labels)}
            print(f"   Mapping original labels to 0-based indices: {label_to_final_index}")
            # Apply mapping
            self.samples = [(img_path, label_to_final_index[label]) for img_path, label in self.samples]
            self.targets = [label_to_final_index[label] for _, label in self.samples] # Store final targets

        # Final check on target range
        if self.targets:
            min_target, max_target = min(self.targets), max(self.targets)
            print(f"   Final target labels range: [{min_target}, {max_target}] for {len(self.classes)} classes.")
            if max_target >= len(self.classes):
                 warnings.warn(f"⚠️ Maximum target label {max_target} is out of bounds for {len(self.classes)} classes!")

        # Compatibility attributes for torchvision datasets/visualization
        self.imgs = self.samples # List of (image_path, final_label) tuples
        self.loader = lambda path: Image.open(path).convert('RGB') # Default image loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = self.loader(img_path)
        except FileNotFoundError:
            warnings.warn(f"Image file not found during __getitem__: {img_path}")
            # Return a dummy image and label or raise error? Let's return None for now.
            # This should ideally be caught by the DataLoader's collate_fn if it happens often.
            return None, -1 # Indicate error
        except Exception as e:
            warnings.warn(f"Error loading image {img_path}: {e}")
            return None, -1

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                 warnings.warn(f"Error applying transform to image {img_path}: {e}")
                 # Return untransformed image or None? Let's return None.
                 return None, label # Return original label with None image

        return image, label

# Example of a collate function to handle None values from __getitem__
def collate_fn_skip_error(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.Tensor(), torch.Tensor() # Return empty tensors if batch is empty
    return torch.utils.data.dataloader.default_collate(batch)

# In main.py, you would use this collate_fn in the DataLoader:
# train_loader = DataLoader(..., collate_fn=collate_fn_skip_error)
# val_loader = DataLoader(..., collate_fn=collate_fn_skip_error)
# test_loader = DataLoader(..., collate_fn=collate_fn_skip_error)
