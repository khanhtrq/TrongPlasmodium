import matplotlib.pyplot as plt
import numpy as np
import torchvision
import random
import torch
import warnings
import math

def imshow(inp, title=None, model_config=None):
    """Displays an image tensor. Handles denormalization using model_config if provided."""
    if not isinstance(inp, torch.Tensor):
        warnings.warn(f"Input to imshow is not a tensor (type: {type(inp)}). Skipping display.")
        return

    inp = inp.cpu().numpy().transpose((1, 2, 0)) # C, H, W -> H, W, C

    # Use mean/std from model_config or default ImageNet values
    mean = np.array(model_config.get('mean', [0.485, 0.456, 0.406])) if model_config else np.array([0.485, 0.456, 0.406])
    std = np.array(model_config.get('std', [0.229, 0.224, 0.225])) if model_config else np.array([0.229, 0.224, 0.225])

    # Denormalize
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1) # Clip values to [0, 1] range

    plt.figure(figsize=(8, 8)) # Slightly larger default size
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=12)
    plt.axis('off')
    plt.show()

def plot_class_distribution_with_ratios(dataset, title="Class Distribution"):
    """Plots and prints the class distribution of a dataset."""
    if not hasattr(dataset, 'targets') or not hasattr(dataset, 'classes'):
        print("⚠️ Cannot plot class distribution: Dataset missing 'targets' or 'classes' attribute.")
        return

    labels = dataset.targets # Use the final mapped labels
    class_names = dataset.classes
    num_classes = len(class_names)
    counts = np.bincount(labels, minlength=num_classes) # Use bincount for efficiency
    total = len(labels)

    print(f"\n📊 {title} (Total: {total} samples)")
    if total == 0:
        print("   Dataset is empty.")
        return

    for i, class_name in enumerate(class_names):
        count = counts[i]
        ratio = count / total * 100 if total > 0 else 0
        print(f" - {class_name} (idx {i}): {count} samples ({ratio:.2f}%)")

    plt.figure(figsize=(max(6, num_classes * 0.6), 5)) # Adjust width based on num classes
    bars = plt.bar(class_names, counts, color='skyblue')
    plt.xlabel("Classes", fontsize=10)
    plt.ylabel("Number of Samples", fontsize=10)
    plt.title(title, fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9) # Rotate labels for better readability
    plt.yticks(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', fontsize=8) # Add text labels

    plt.tight_layout()
    plt.show()

def analyze_class_distribution_across_splits(datasets_dict):
    """
    Analyzes and prints the class distribution across different dataset splits (e.g., train, val, test).

    Args:
        datasets_dict (dict): Dictionary where keys are split names (str) and values are Dataset objects.
                              Example: {'Train': train_dataset, 'Validation': val_dataset, 'Test': test_dataset}
    """
    print("\n📊 Analyzing Class Distribution Across Splits:")
    if not datasets_dict:
        print("   No datasets provided.")
        return

    # Get class names from the first dataset (assuming they are consistent)
    first_dataset_name = next(iter(datasets_dict))
    first_dataset = datasets_dict[first_dataset_name]
    if not hasattr(first_dataset, 'classes') or not first_dataset.classes:
        print("   Could not determine class names from the first dataset.")
        return
    class_names = first_dataset.classes
    num_classes = len(class_names)

    # Initialize counts
    class_counts = {name: {split: 0 for split in datasets_dict} for name in class_names}
    split_totals = {split: 0 for split in datasets_dict}

    # Populate counts
    for split_name, dataset in datasets_dict.items():
        if not hasattr(dataset, 'targets'):
            print(f"   ⚠️ Skipping split '{split_name}': Missing 'targets' attribute.")
            continue
        if not hasattr(dataset, 'classes') or dataset.classes != class_names:
             print(f"   ⚠️ Skipping split '{split_name}': Class names mismatch or missing.")
             continue

        labels = dataset.targets
        split_totals[split_name] = len(labels)
        counts_this_split = np.bincount(labels, minlength=num_classes)

        for i, class_name in enumerate(class_names):
            class_counts[class_name][split_name] = counts_this_split[i]

    # Calculate overall totals per class
    total_per_class = {name: sum(counts[split] for split in datasets_dict) for name, counts in class_counts.items()}

    # Print results
    print("\n--- Absolute Counts per Split ---")
    header = f"{'Class':<15}" + "".join([f"{split:>12}" for split in datasets_dict]) + f"{'Total':>12}"
    print(header)
    print("-" * len(header))
    for i, class_name in enumerate(class_names):
        row = f"{class_name:<15}"
        for split_name in datasets_dict:
            row += f"{class_counts[class_name][split_name]:>12}"
        row += f"{total_per_class[class_name]:>12}"
        print(row)
    print("-" * len(header))
    footer = f"{'Total':<15}" + "".join([f"{split_totals[split]:>12}" for split in datasets_dict]) + f"{sum(split_totals.values()):>12}"
    print(footer)


    print("\n--- Percentage within each Split ---")
    header = f"{'Class':<15}" + "".join([f"{split+' (%)':>12}" for split in datasets_dict])
    print(header)
    print("-" * len(header))
    for i, class_name in enumerate(class_names):
        row = f"{class_name:<15}"
        for split_name in datasets_dict:
            total_split = split_totals[split_name]
            ratio = (class_counts[class_name][split_name] / total_split * 100) if total_split > 0 else 0
            row += f"{ratio:>12.2f}"
        print(row)
    print("-" * len(header))

    print("\n--- Percentage of Class across Splits ---")
    header = f"{'Class':<15}" + "".join([f"{split+' (%)':>12}" for split in datasets_dict])
    print(header)
    print("-" * len(header))
    for i, class_name in enumerate(class_names):
        row = f"{class_name:<15}"
        total_class = total_per_class[class_name]
        for split_name in datasets_dict:
            ratio = (class_counts[class_name][split_name] / total_class * 100) if total_class > 0 else 0
            row += f"{ratio:>12.2f}"
        print(row)
    print("-" * len(header))


def imshow_helper(inp, title=None, ax=None, model_config=None):
    """Helper to display image tensor on a given matplotlib Axes object."""
    if not isinstance(inp, torch.Tensor):
        warnings.warn(f"Input to imshow_helper is not a tensor (type: {type(inp)}). Skipping.")
        if ax: ax.axis('off')
        return

    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array(model_config.get('mean', [0.485, 0.456, 0.406])) if model_config else np.array([0.485, 0.456, 0.406])
    std = np.array(model_config.get('std', [0.229, 0.224, 0.225])) if model_config else np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    if ax is None:
        fig, ax = plt.subplots() # Create new figure if no axes provided

    ax.imshow(inp)
    if title is not None:
        ax.set_title(title, fontsize=9)
    ax.axis('off')

def plot_sample_images_per_class(dataset, num_samples=5, model_config=None):
    """Plots a grid of sample images for each class in the dataset."""
    if not hasattr(dataset, 'classes') or not dataset.classes:
        print("⚠️ Cannot plot samples: Dataset missing 'classes' attribute.")
        return
    if not hasattr(dataset, 'imgs') or not hasattr(dataset, 'loader') or not hasattr(dataset, 'transform'):
         print("⚠️ Cannot plot samples: Dataset missing 'imgs', 'loader', or 'transform'.")
         return

    class_names = dataset.classes
    num_classes = len(class_names)
    if num_classes == 0:
        print("⚠️ No classes found in the dataset.")
        return

    samples_per_class = {i: [] for i in range(num_classes)} # Use class index as key
    img_paths_labels = dataset.imgs # List of (path, label_index)

    # Shuffle to get random samples if dataset is large
    indices = list(range(len(img_paths_labels)))
    random.shuffle(indices)

    found_samples = 0
    total_target_samples = num_classes * num_samples

    print(f"\n🖼️ Plotting {num_samples} sample images per class...")

    for idx in indices:
        img_path, label = img_paths_labels[idx]
        if len(samples_per_class[label]) < num_samples:
            try:
                image = dataset.loader(img_path) # Load PIL image
                if dataset.transform is not None:
                    # Apply transform to get the tensor as used by the model
                    img_tensor = dataset.transform(image)
                else:
                    # Minimal transform if none provided
                    img_tensor = transforms.ToTensor()(image)

                # Ensure it's a tensor
                if isinstance(img_tensor, torch.Tensor):
                    samples_per_class[label].append(img_tensor.detach().cpu()) # Store tensor on CPU
                    found_samples += 1
                else:
                     warnings.warn(f"Transformed image for {img_path} is not a tensor. Skipping.")

            except FileNotFoundError:
                 warnings.warn(f"Sample image not found: {img_path}. Skipping.")
                 continue
            except Exception as e:
                warnings.warn(f"Could not load/transform sample image {img_path}: {e}. Skipping.")
                continue

        # Check if we have enough samples for all classes
        if found_samples >= total_target_samples:
            break

    # --- Plotting ---
    # Adjust figure size dynamically
    fig_width = num_samples * 2.5
    fig_height = num_classes * 2.5
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(fig_width, fig_height), squeeze=False)
    fig.suptitle("Sample Images per Class", fontsize=14)

    for i, class_name in enumerate(class_names):
        axes[i, 0].set_ylabel(class_name, rotation=90, size='large', labelpad=10)
        class_samples = samples_per_class[i]
        for j in range(num_samples):
            ax = axes[i, j]
            if j < len(class_samples):
                img_tensor = class_samples[j]
                # Use imshow_helper for consistent display and denormalization
                imshow_helper(img_tensor, ax=ax, model_config=model_config)
            else:
                # If fewer than num_samples found for a class
                ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', fontsize=10)
                ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plt.show()


def smooth_curve(points, factor=0.6):
    """Applies exponential moving average smoothing to a list of points."""
    smoothed_points = []
    if not points: # Handle empty list
        return []
    for point in points:
        # Convert tensors to numpy floats if necessary
        if isinstance(point, torch.Tensor):
            point = point.detach().cpu().item() # Use .item() for scalar tensors
        elif not isinstance(point, (int, float, np.number)):
             # Try converting potential numpy arrays/scalars
             try:
                 point = float(point)
             except (ValueError, TypeError):
                 warnings.warn(f"Cannot convert point of type {type(point)} to float for smoothing. Skipping point.")
                 # Append previous smoothed value or NaN? Let's append NaN.
                 smoothed_points.append(float('nan'))
                 continue

        # Handle NaN values in input - propagate them
        if math.isnan(point):
             smoothed_points.append(float('nan'))
             continue

        if smoothed_points and not math.isnan(smoothed_points[-1]):
            # Apply smoothing formula only if previous point was valid
            smoothed_points.append(smoothed_points[-1] * factor + point * (1 - factor))
        else:
            # First point, or previous point was NaN
            smoothed_points.append(point)
    return smoothed_points

def plot_training_curves(history, title_suffix='', save_path=None, smoothing_factor=0.6):
    """
    Plots training and validation loss, accuracy (macro and weighted),
    and validation precision, recall, F1 (macro) curves.

    Args:
        history (dict): Dictionary containing training history lists
                        (e.g., 'train_loss', 'val_loss', 'train_acc_macro', etc.).
        title_suffix (str): Suffix to add to plot titles.
        save_path (str, optional): Path to save the combined plot figure. If None, displays the plot.
        smoothing_factor (float): Factor for exponential moving average smoothing (0.0 to 1.0).
                                  Set to 0.0 to disable smoothing.
    """
    required_keys = [
        'train_loss', 'val_loss',
        'train_acc_macro', 'val_acc_macro',
        'train_acc_weighted', 'val_acc_weighted',
        'val_precision_macro', 'val_recall_macro', 'val_f1_macro'
        # Optional: 'val_precision_weighted', 'val_recall_weighted', 'val_f1_weighted'
    ]

    # Check if all required keys are present
    if not all(key in history for key in required_keys):
        print("⚠️ Cannot plot training curves: History dictionary is missing required keys.")
        print(f"   Required: {required_keys}")
        print(f"   Available: {list(history.keys())}")
        return

    # Determine number of epochs from a reliable key
    num_epochs = len(history['train_loss'])
    epochs = range(1, num_epochs + 1)

    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
    fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # Create 3 subplots
    fig.suptitle(f'Training Curves {title_suffix}', fontsize=16)

    # --- Plot 1: Loss ---
    ax1 = axes[0]
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    if smoothing_factor > 0:
        ax1.plot(epochs, smooth_curve(train_loss, smoothing_factor), label='Train Loss (Smoothed)', alpha=0.8)
        ax1.plot(epochs, smooth_curve(val_loss, smoothing_factor), label='Val Loss (Smoothed)', alpha=0.8)
        ax1.plot(epochs, train_loss, label='Train Loss (Raw)', alpha=0.3, linestyle=':')
        ax1.plot(epochs, val_loss, label='Val Loss (Raw)', alpha=0.3, linestyle=':')
    else:
        ax1.plot(epochs, train_loss, label='Train Loss')
        ax1.plot(epochs, val_loss, label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    # ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Accuracy ---
    ax2 = axes[1]
    train_acc_m = history['train_acc_macro']
    val_acc_m = history['val_acc_macro']
    train_acc_w = history['train_acc_weighted']
    val_acc_w = history['val_acc_weighted']
    if smoothing_factor > 0:
        ax2.plot(epochs, smooth_curve(train_acc_m, smoothing_factor), label='Train Acc (Macro, Smooth)', alpha=0.8)
        ax2.plot(epochs, smooth_curve(val_acc_m, smoothing_factor), label='Val Acc (Macro, Smooth)', alpha=0.8)
        ax2.plot(epochs, smooth_curve(train_acc_w, smoothing_factor), label='Train Acc (Weighted, Smooth)', alpha=0.8, linestyle='--')
        ax2.plot(epochs, smooth_curve(val_acc_w, smoothing_factor), label='Val Acc (Weighted, Smooth)', alpha=0.8, linestyle='--')
        # Optional: Plot raw data lightly
        # ax2.plot(epochs, train_acc_m, alpha=0.2, linestyle=':')
        # ax2.plot(epochs, val_acc_m, alpha=0.2, linestyle=':')
    else:
        ax2.plot(epochs, train_acc_m, label='Train Acc (Macro)')
        ax2.plot(epochs, val_acc_m, label='Val Acc (Macro)')
        ax2.plot(epochs, train_acc_w, label='Train Acc (Weighted)', linestyle='--')
        ax2.plot(epochs, val_acc_w, label='Val Acc (Weighted)', linestyle='--')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    # ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(bottom=max(0, ax2.get_ylim()[0]), top=min(1.05, ax2.get_ylim()[1])) # Set sensible y-limits for accuracy


    # --- Plot 3: Validation Precision, Recall, F1 (Macro) ---
    ax3 = axes[2]
    val_p = history['val_precision_macro']
    val_r = history['val_recall_macro']
    val_f1 = history['val_f1_macro']
    if smoothing_factor > 0:
        ax3.plot(epochs, smooth_curve(val_p, smoothing_factor), label='Val Precision (Macro, Smooth)', alpha=0.8)
        ax3.plot(epochs, smooth_curve(val_r, smoothing_factor), label='Val Recall (Macro, Smooth)', alpha=0.8)
        ax3.plot(epochs, smooth_curve(val_f1, smoothing_factor), label='Val F1 (Macro, Smooth)', alpha=0.8)
        # Optional: Plot raw data lightly
        # ax3.plot(epochs, val_p, alpha=0.2, linestyle=':')
        # ax3.plot(epochs, val_r, alpha=0.2, linestyle=':')
        # ax3.plot(epochs, val_f1, alpha=0.2, linestyle=':')
    else:
        ax3.plot(epochs, val_p, label='Val Precision (Macro)')
        ax3.plot(epochs, val_r, label='Val Recall (Macro)')
        ax3.plot(epochs, val_f1, label='Val F1 (Macro)')
    ax3.set_title('Validation P/R/F1 (Macro)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    # ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_ylim(bottom=max(0, ax3.get_ylim()[0]), top=min(1.05, ax3.get_ylim()[1])) # Set sensible y-limits for scores

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

    # Save the figure if a path is provided, otherwise display it
    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            print(f"📈 Training curves plot saved to {save_path}")
            plt.close(fig) # Close the figure after saving
        except Exception as e:
            print(f"⚠️ Could not save training curves plot to {save_path}: {e}")
            plt.show() # Show if saving failed
    else:
        plt.show()
