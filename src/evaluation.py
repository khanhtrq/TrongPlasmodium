import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def infer_from_annotation(model, annotation_file, class_names, root_dir, device, transform=None, input_size=(224, 224), dataloader=None):
    """
    Run inference using either a dataloader or by loading images from annotation file.
    
    Args:
        model: The trained model
        annotation_file: Path to annotation file (used only if dataloader is None)
        class_names: List of class names
        root_dir: Root directory for images (used only if dataloader is None)
        device: Device to run inference on
        transform: Transform to apply to images (used only if dataloader is None)
        input_size: Input size for images (used only if dataloader is None)
        dataloader: DataLoader to use for inference (if provided, annotation_file will be ignored)
    """
    y_true = []
    y_pred = []
    
    if dataloader is not None:
        # Use the provided dataloader for inference - much faster!
        print("🚀 Using DataLoader for batch inference")
        
        for inputs, labels in tqdm(dataloader, desc="Inferencing batches"):
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())
            
        print(f"✅ Completed inference on {len(y_true)} samples using batched processing")
    else:
        # Fallback to loading from annotation file (slower)
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            print("⚠️ No transform provided, using default transform for inference")
        else:
            print("✅ Using provided transform for inference (matching training transform)")

        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Inferencing individual images"):
            rel_path, label = line.strip().split()
            img_path = os.path.join(root_dir, rel_path)
            label = int(label)

            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                preds = outputs.argmax(dim=1)

            y_pred.append(preds.item())
            y_true.append(label)

    return y_true, y_pred


def report_classification(y_true, y_pred, class_names, save_path_base=None):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred, normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Oranges', ax=ax, xticks_rotation=45)
    plt.title('Confusion Matrix (Raw Counts)')
    if save_path_base:
        raw_cm_path = save_path_base + '_raw.png'
        plt.savefig(raw_cm_path)
    plt.show()

    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp_norm.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    plt.title('Normalized Confusion Matrix')
    if save_path_base:
        norm_cm_path = save_path_base + '_normalized.png'
        plt.savefig(norm_cm_path)
    plt.show()
