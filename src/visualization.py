import matplotlib.pyplot as plt
import numpy as np
import torchvision
import random
import torch

def imshow(inp, title=None):
    """Hiển thị ảnh từ tensor trong notebook.
    
    Nếu có biến normalized, bạn có thể cần reverse normalization ở đây.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean

    plt.figure(figsize=(12, 6))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

def plot_class_distribution_with_ratios(dataset, title="Class Distribution"):
    labels = [label for _, label in dataset.imgs]
    class_names = dataset.classes
    counts = [labels.count(i) for i in range(len(class_names))]
    total = sum(counts)

    print(f"\n📊 {title}")
    for i, class_name in enumerate(class_names):
        ratio = counts[i] / total * 100 if total > 0 else 0
        print(f" - {class_name}: {counts[i]} ảnh ({ratio:.2f}%)")

    plt.figure(figsize=(8,6))
    plt.bar(class_names, counts, color='skyblue')
    plt.xlabel("Classes")
    plt.ylabel("Số lượng ảnh")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_class_distribution_across_splits(train_dataset, val_dataset, test_dataset):
    datasets = {
        'Train': train_dataset,
        'Validation': val_dataset,
        'Test': test_dataset
    }
    
    class_names = train_dataset.classes
    class_counts = {name: {'Train':0, 'Validation':0, 'Test':0} for name in class_names}
    
    for split_name, dataset in datasets.items():
        labels = [label for _, label in dataset.imgs]
        for label in labels:
            class_name = class_names[label]
            class_counts[class_name][split_name] += 1

    total_per_class = {name: sum(counts.values()) for name, counts in class_counts.items()}

    print("\n📊 Phân bố class theo các tập:")
    for class_name in class_names:
        total = total_per_class[class_name]
        if total == 0:
            continue
        train_ratio = class_counts[class_name]['Train'] / total * 100
        val_ratio = class_counts[class_name]['Validation'] / total * 100
        test_ratio = class_counts[class_name]['Test'] / total * 100
        
        print(f"- {class_name}: {train_ratio:.2f}% Train, {val_ratio:.2f}% Validation, {test_ratio:.2f}% Test")

def imshow_helper(inp, title=None):
    """Helper to display image tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')

def plot_sample_images_per_class(dataset, num_samples=5):
    """Plots a grid of sample images for each class in the dataset."""
    class_names = dataset.classes
    samples_per_class = {class_name: [] for class_name in class_names}
    img_paths_labels = dataset.imgs

    for img_path, label in img_paths_labels:
        class_name = class_names[label]
        if len(samples_per_class[class_name]) < num_samples:
            try:
                image = dataset.loader(img_path)
                if dataset.transform is not None:
                    image = dataset.transform(image)
                samples_per_class[class_name].append(image)
            except Exception as e:
                print(f"Could not load/transform image {img_path}: {e}")
                continue

        if all(len(samples) >= num_samples for samples in samples_per_class.values()):
            break

    num_classes = len(class_names)
    if num_classes == 0:
        print("No classes found in the dataset.")
        return

    fig_width = num_samples * 2
    fig_height = num_classes * 2
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(fig_width, fig_height), squeeze=False)

    for i, class_name in enumerate(class_names):
        axes[i, 0].set_ylabel(class_name, rotation=90, size='large')
        for j in range(num_samples):
            ax = axes[i, j]
            if j < len(samples_per_class[class_name]):
                img_tensor = samples_per_class[class_name][j]
                if isinstance(img_tensor, torch.Tensor):
                    img_tensor = img_tensor.detach() if img_tensor.requires_grad else img_tensor
                    img_tensor = img_tensor.cpu()
                    plt.sca(ax)
                    imshow_helper(img_tensor)
                else:
                    ax.imshow(img_tensor)
                    ax.axis('off')
            else:
                ax.axis('off')

    plt.tight_layout()
    plt.show()

def smooth_curve(points, factor=0.6):
    smoothed_points = []
    for point in points:
        if isinstance(point, torch.Tensor):
            point = point.cpu().numpy()
        if smoothed_points:
            smoothed_points.append(smoothed_points[-1] * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_training_curves(history, title_suffix=''):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(smooth_curve(history['train_loss']), label='Train Loss')
    plt.plot(smooth_curve(history['val_loss']), label='Val Loss')
    plt.title('Loss Curve ' + title_suffix)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 3, 2)
    plt.plot(smooth_curve(history['train_acc_macro']), label='Train Macro Accuracy')
    plt.plot(smooth_curve(history['val_acc_macro']), label='Val Macro Accuracy')
    plt.plot(smooth_curve(history['train_acc_weighted']), label='Train Weighted Accuracy')
    plt.plot(smooth_curve(history['val_acc_weighted']), label='Val Weighted Accuracy')
    plt.title('Accuracy Curve ' + title_suffix)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 3, 3)
    plt.plot(smooth_curve(history['val_precision']), label='Val Precision')
    plt.plot(smooth_curve(history['val_recall']), label='Val Recall')
    plt.plot(smooth_curve(history['val_f1']), label='Val F1-Score')
    plt.title('Precision, Recall, F1 Curve ' + title_suffix)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
