import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset  # Import ConcatDataset
from PIL import Image
from torchvision import transforms, datasets  # Import datasets
import numpy as np
import warnings


class AnnotationDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None, class_names=None, class_remapping=None):
        """
        Args:
            annotation_file (str): Path to the annotation file (e.g., 'train_annotation.txt').
                                   Each line: relative/path/to/image.jpg label_index
            root_dir (str): Root directory where images are located.
            transform (callable, optional): Optional transform to be applied on a sample.
            class_names (list, optional): List of class names in the desired order.
                                          If provided, labels in the annotation file will be
                                          mapped to indices corresponding to this list.
            class_remapping (dict, optional): Dictionary for class remapping config.
                                            Expected keys: 'enabled', 'mapping', 'final_class_names'
        """
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        self.original_labels = set()  # Store labels as read from the file
        self.targets = []  # Final (potentially remapped) label indices
        self.class_remapping = class_remapping

        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ Annotation file not found: {annotation_file}")

        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if not stripped_line:  # Skip empty lines
                continue

            parts = stripped_line.split()

            if len(parts) < 2:  # Must have at least a path-like part and a label part
                warnings.warn(
                    f"⚠️ Skipping malformed line {i+1} in {annotation_file} (expected path and label): '{stripped_line}'")
                continue

            # Assume the last part is the label, everything else is the path
            label_str = parts[-1]
            path = " ".join(parts[:-1])

            if not path:  # Handle cases where path might become empty if line was just " label"
                warnings.warn(
                    f"⚠️ Skipping line {i+1} with missing path in {annotation_file}: '{stripped_line}'")
                continue

            try:
                label = int(label_str)
            except ValueError:
                warnings.warn(
                    f"⚠️ Skipping line {i+1} with non-integer label ('{label_str}') in {annotation_file}: '{stripped_line}'")
                continue

            full_path = os.path.join(self.root_dir, path)
            self.samples.append((full_path, label))
            self.original_labels.add(label)

        if not self.samples:
            raise ValueError(
                f"❌ No valid samples loaded from {annotation_file}. Check file format and paths.")


        # --- Apply Class Remapping (NEW) ---
        if self.class_remapping and self.class_remapping.get('enabled', False):
            self._apply_class_remapping()

        # --- Class Name and Label Mapping Logic ---
        unique_original_labels = sorted(list(self.original_labels))

        if class_names is not None:
            # Use remapped class names if available, otherwise use provided class_names
            if self.class_remapping and self.class_remapping.get('enabled', False) and self.class_remapping.get('final_class_names'):
                self.classes = self.class_remapping['final_class_names']
            else:
                self.classes = class_names

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
                label_to_final_index = {
                    label: label for label in unique_original_labels}

            else:
                # Assume the sorted unique labels correspond to the order of class_names
                label_to_final_index = {
                    orig_label: idx for idx, orig_label in enumerate(unique_original_labels)}

            # Apply the mapping
            try:
                remapped_samples = []
                self.targets = []
                for img_path, orig_label in self.samples:
                    if orig_label not in label_to_final_index:
                        warnings.warn(
                            f"⚠️ Original label {orig_label} (from {img_path}) not found in mapping keys {list(label_to_final_index.keys())}. Skipping sample.")
                        continue
                    final_label = label_to_final_index[orig_label]
                    if not (0 <= final_label < num_expected_classes):
                        warnings.warn(
                            f"⚠️ Mapped label {final_label} (from original {orig_label}) is outside the expected range [0, {num_expected_classes-1}] for class_names. Skipping sample.")
                        continue
                    remapped_samples.append((img_path, final_label))
                    self.targets.append(final_label)
                self.samples = remapped_samples
                if not self.samples:
                    raise ValueError(
                        "❌ No samples remained after label remapping. Check consistency between annotation file labels and provided class_names.")

            except KeyError as e:
                raise ValueError(
                    f"❌ Error remapping labels. Original label {e} from annotation file not found in the derived mapping. Ensure consistency.")

        else:
            # Infer class names from sorted unique labels
            self.classes = [str(i) for i in unique_original_labels]
            num_expected_classes = len(self.classes)
            # Create a mapping from original label to a 0-based contiguous index
            label_to_final_index = {orig_label: idx for idx,
                                    orig_label in enumerate(unique_original_labels)}
            # Apply mapping
            self.samples = [(img_path, label_to_final_index[label])
                            for img_path, label in self.samples]
            self.targets = [label_to_final_index[label]
                            for _, label in self.samples]  # Store final targets

        # Final check on target range
        if self.targets:
            min_target, max_target = min(self.targets), max(self.targets)

            if max_target >= len(self.classes):
                warnings.warn(
                    f"⚠️ Maximum target label {max_target} is out of bounds for {len(self.classes)} classes!")

        # Compatibility attributes for torchvision datasets/visualization
        self.imgs = self.samples  # List of (image_path, final_label) tuples
        self.loader = lambda path: Image.open(
            path).convert('RGB')  # Default image loader

    def _apply_class_remapping(self):
        """Apply class remapping to samples and update original_labels."""
        mapping = self.class_remapping.get('mapping', {})
        if not mapping:
            return


        # Apply remapping to samples
        remapped_samples = []
        excluded_samples = []
        new_original_labels = set()

        for img_path, label in self.samples:
            # Apply remapping if label exists in mapping
            new_label = mapping.get(label, label)

            # Check if label should be excluded from training (-1)
            if new_label == -1:
                excluded_samples.append((img_path, label))
                continue

            remapped_samples.append((img_path, new_label))
            new_original_labels.add(new_label)

        self.samples = remapped_samples
        self.original_labels = new_original_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = self.loader(img_path)
        except FileNotFoundError:
            warnings.warn(
                f"Image file not found during __getitem__: {img_path}")
            # Return a dummy image and label or raise error? Let's return None for now.
            # This should ideally be caught by the DataLoader's collate_fn if it happens often.
            return None, -1  # Indicate error
        except Exception as e:
            warnings.warn(f"Error loading image {img_path}: {e}")
            return None, -1

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                warnings.warn(
                    f"Error applying transform to image {img_path}: {e}")
                # Return untransformed image or None? Let's return None.
                return None, label  # Return original label with None image

        return image, label

# --- New ImageFolder Wrapper ---


class ImageFolderWrapper(datasets.ImageFolder):
    """
    A wrapper around torchvision.datasets.ImageFolder to provide
    consistent attributes with AnnotationDataset (e.g., .imgs, .loader).
    """

    def __init__(self, root, transform=None, target_transform=None, loader=datasets.folder.default_loader, is_valid_file=None, class_remapping=None):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)

        self.class_remapping = class_remapping

        # Apply class remapping if enabled
        if self.class_remapping and self.class_remapping.get('enabled', False):
            self._apply_class_remapping()

        # Add compatibility attributes
        self.imgs = self.samples  # Alias for compatibility

        if not self.samples:
            warnings.warn(
                f"⚠️ No image files found in {root}. Check the directory structure and image extensions.")
        else:
            min_target, max_target = min(self.targets), max(self.targets)

    def _apply_class_remapping(self):
        """Apply class remapping to ImageFolder samples and targets."""
        mapping = self.class_remapping.get('mapping', {})
        if not mapping:
            return


        # Apply remapping to samples and targets
        remapped_samples = []
        remapped_targets = []
        excluded_samples = []

        for (img_path, label), target in zip(self.samples, self.targets):
            new_label = mapping.get(label, label)
            new_target = mapping.get(target, target)

            # Check if should be excluded from training (-1)
            if new_label == -1 or new_target == -1:
                excluded_samples.append((img_path, label))
                continue

            remapped_samples.append((img_path, new_label))
            remapped_targets.append(new_target)

        self.samples = remapped_samples
        self.targets = remapped_targets


        if excluded_samples:
            excluded_classes = sorted(
                set(orig_label for _, orig_label in excluded_samples))

        # Update class names if provided
        if self.class_remapping.get('final_class_names'):
            # Get unique remapped labels to determine new classes
            unique_labels = sorted(set(self.targets))
            if len(unique_labels) == len(self.class_remapping['final_class_names']):
                self.classes = self.class_remapping['final_class_names']


# --- New Combined Dataset Wrapper ---


class CombinedDataset(Dataset):
    """
    Wraps ConcatDataset to provide combined 'targets' and 'classes' attributes
    for compatibility with analysis and visualization functions.
    """

    def __init__(self, datasets):
        if not datasets:
            raise ValueError(
                "Cannot create CombinedDataset from an empty list of datasets.")

        self.concat_dataset = ConcatDataset(datasets)
        self.datasets = datasets

        # --- Combine classes and targets ---
        first_dataset = datasets[0]
        if not hasattr(first_dataset, 'classes'):
            raise AttributeError(
                "The first dataset in the list must have a 'classes' attribute.")
        self.classes = first_dataset.classes

        all_targets = []
        for i, ds in enumerate(datasets):
            if not hasattr(ds, 'classes') or ds.classes != self.classes:
                warnings.warn(
                    f"Dataset {i} has missing or inconsistent 'classes' attribute. Skipping its targets.")
                continue  # Or raise error if strict consistency is needed
            if hasattr(ds, 'targets'):
                all_targets.extend(ds.targets)
            else:
                warnings.warn(
                    f"Dataset {i} is missing 'targets' attribute. Cannot combine targets.")
                # If targets are essential, raise an error here instead
                self.targets = []  # Mark targets as unavailable
                break
        else:  # Only runs if the loop completes without break
            self.targets = all_targets


        # --- Add other compatibility attributes (optional, may need refinement) ---
        # Use attributes from the first dataset as representative
        if hasattr(first_dataset, 'imgs'):
            # Note: This 'imgs' won't directly map to indices in the combined dataset easily.
            # It's mainly for functions that might expect the attribute to exist.
            self.imgs = first_dataset.imgs
        if hasattr(first_dataset, 'loader'):
            self.loader = first_dataset.loader

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]

# Example of a collate function to handle None values from __getitem__


def collate_fn_skip_error(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.Tensor(), torch.Tensor()  # Return empty tensors if batch is empty
    return torch.utils.data.dataloader.default_collate(batch)

# In main.py, you would use this collate_fn in the DataLoader:
# train_loader = DataLoader(..., collate_fn=collate_fn_skip_error)
# val_loader = DataLoader(..., collate_fn=collate_fn_skip_error)
# test_loader = DataLoader(..., collate_fn=collate_fn_skip_error)

# === WeightedRandomSampler Utilities ===


def compute_class_weights_from_dataset(dataset, num_classes, weight_calculation='inverse', apply_sqrt=False, min_weight=0.1, max_weight=10.0):
    """
    Compute class weights from a dataset for WeightedRandomSampler.

    Args:
        dataset: Dataset with 'targets' attribute containing class indices
        num_classes (int): Number of classes
        weight_calculation (str): Method to calculate weights
            - 'inverse': 1.0 / class_count
            - 'balanced': sklearn-style balanced weights
            - 'custom': Use predefined custom weights (not implemented here)
        apply_sqrt (bool): Apply square root to weights for softer balancing
        min_weight (float): Minimum weight value to prevent extreme weights
        max_weight (float): Maximum weight value to prevent extreme weights

    Returns:
        torch.Tensor: Weights for each class
    """
    import torch
    import numpy as np
    from collections import Counter


    # Get targets from dataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        elif isinstance(targets, list):
            targets = np.array(targets)
    else:
        raise AttributeError(
            "Dataset must have 'targets' attribute for WeightedRandomSampler")

    # Filter out any -1 labels that might have slipped through
    valid_targets = targets[targets != -1]
    if len(valid_targets) != len(targets):
        targets = valid_targets

    # Count class frequencies
    class_counts = Counter(targets)

    # Ensure all classes are represented (fill missing classes with 1)
    for class_idx in range(num_classes):
        if class_idx not in class_counts:
            class_counts[class_idx] = 1


    # Calculate weights based on method
    if weight_calculation == 'inverse':
        # Simple inverse frequency
        class_weights = torch.zeros(num_classes)
        total_samples = sum(class_counts.values())
        for class_idx in range(num_classes):
            class_weights[class_idx] = total_samples / class_counts[class_idx]

    elif weight_calculation == 'balanced':
        # Sklearn-style balanced weights: n_samples / (n_classes * count_for_class)
        n_samples = len(targets)
        class_weights = torch.zeros(num_classes)
        for class_idx in range(num_classes):
            class_weights[class_idx] = n_samples / \
                (num_classes * class_counts[class_idx])

    else:
        raise ValueError(
            f"Unsupported weight_calculation method: {weight_calculation}")

    # Apply square root for softer balancing if requested
    if apply_sqrt:
        class_weights = torch.sqrt(class_weights)

    # Clamp weights to prevent extreme values
    class_weights = torch.clamp(class_weights, min=min_weight, max=max_weight)

    return class_weights


def create_weighted_random_sampler(dataset, num_classes, sampler_config):
    """
    Create a WeightedRandomSampler based on dataset and configuration.

    Args:
        dataset: Dataset with 'targets' attribute
        num_classes (int): Number of classes
        sampler_config (dict): Configuration for the sampler

    Returns:
        torch.utils.data.WeightedRandomSampler or None if disabled
    """
    from torch.utils.data import WeightedRandomSampler

    if not sampler_config.get('enabled', False):
        return None


    # Compute class weights
    class_weights = compute_class_weights_from_dataset(
        dataset,
        num_classes,
        weight_calculation=sampler_config.get('weight_calculation', 'inverse'),
        apply_sqrt=sampler_config.get('apply_sqrt', False),
        min_weight=sampler_config.get('min_weight', 0.1),
        max_weight=sampler_config.get('max_weight', 10.0)
    )

    # Get targets from dataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        elif isinstance(targets, list):
            targets = np.array(targets)
    else:
        raise AttributeError(
            "Dataset must have 'targets' attribute for WeightedRandomSampler")

    # Filter out samples with -1 labels (should already be filtered, but double-check)
    valid_indices = [i for i, target in enumerate(targets) if target != -1]
    if len(valid_indices) != len(targets):
        targets = targets[valid_indices]

    # Create sample weights (weight for each sample based on its class)
    sample_weights = torch.zeros(len(targets))
    for idx, target in enumerate(targets):
        if target >= 0 and target < len(class_weights):  # Safety check
            sample_weights[idx] = class_weights[target]
        else:
            sample_weights[idx] = 0.0

    # Create the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(targets),  # Use filtered length
        replacement=sampler_config.get('replacement', True)
    )
    return sampler


def get_effective_sampler_config(main_config, phase_config=None):
    """
    Get the effective sampler configuration, with phase-specific config overriding main config.

    Args:
        main_config (dict): Main WeightedRandomSampler configuration
        phase_config (dict, optional): Phase-specific configuration (e.g., for classifier training)

    Returns:
        dict: Effective configuration to use
    """
    if phase_config is None:
        return main_config.copy()

    # Start with main config
    effective_config = main_config.copy()

    # Override with phase-specific settings
    for key, value in phase_config.items():
        effective_config[key] = value

    return effective_config
