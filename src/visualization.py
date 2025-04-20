import matplotlib.pyplot as plt
import numpy as np
import torchvision

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

import random
import matplotlib.pyplot as plt
import numpy as np

def imshow(ax, image, title=None):
    npimg = image.numpy().transpose((1, 2, 0))
    ax.imshow(npimg)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=8)

num_samples = 5
class_names = train_dataset.classes
samples_per_class = {class_name: [] for class_name in class_names}

for img_path, label in train_dataset.imgs:
    class_name = class_names[label]
    if len(samples_per_class[class_name]) < num_samples:
        image = train_dataset.loader(img_path)
        if train_dataset.transform is not None:
            image = train_dataset.transform(image)
        samples_per_class[class_name].append(image)
    if all(len(samples) >= num_samples for samples in samples_per_class.values()):
        break

num_classes = len(class_names)
fig, axes = plt.subplots(num_classes, num_samples, figsize=(num_samples*2, num_classes*2))

if num_classes == 1:
    axes = np.expand_dims(axes, 0)
if num_samples == 1:
    axes = np.expand_dims(axes, 1)

for i, class_name in enumerate(class_names):
    for j in range(num_samples):
        ax = axes[i, j]
        if j < len(samples_per_class[class_name]):
            imshow(ax, samples_per_class[class_name][j])
        else:
            ax.axis('off')

import matplotlib.pyplot as plt
import numpy as np

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
