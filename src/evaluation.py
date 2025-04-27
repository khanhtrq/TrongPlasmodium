import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def infer_from_annotation(model, class_names, device, dataloader=None, annotation_file=None, root_dir=None, transform=None, input_size=(224, 224)):
    """
    Run inference on a dataset, prioritizing DataLoader for efficiency.

    Args:
        model: The trained model, already moved to the correct device.
        class_names: List of class names.
        device: Device to run inference on ('cuda' or 'cpu').
        dataloader: (Preferred) DataLoader for the dataset to evaluate.
        annotation_file: (Fallback) Path to annotation file if dataloader is None.
        root_dir: (Fallback) Root directory for images if dataloader is None.
        transform: (Fallback) Transform to apply if dataloader is None.
        input_size: (Fallback) Input size if dataloader is None.

    Returns:
        tuple: (y_true, y_pred) - Lists of true labels and predicted labels.
    """
    model.eval() # Ensure model is in evaluation mode
    y_true = []
    y_pred = []

    if dataloader is not None:
        # --- Preferred Method: Use DataLoader for Batch Inference ---
        print("🚀 Using DataLoader for efficient batch inference...")
        num_samples = len(dataloader.dataset)
        with torch.no_grad(): # Disable gradient calculations for inference
            for inputs, labels in tqdm(dataloader, desc="Inferencing batches", total=len(dataloader)):
                inputs = inputs.to(device)
                # No need for AMP autocast during inference usually, unless model requires it
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)

                y_pred.extend(preds.cpu().numpy())
                y_true.extend(labels.cpu().numpy()) # Assuming labels are on CPU from DataLoader

        print(f"✅ Completed batch inference on {len(y_true)}/{num_samples} samples.")

    elif annotation_file and root_dir and transform:
        # --- Fallback Method: Load Images Individually ---
        print("⚠️ DataLoader not provided. Falling back to slower single-image inference from annotation file.")
        print(f"   Annotation: {annotation_file}")
        print(f"   Root Dir: {root_dir}")

        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"❌ Error: Annotation file not found at {annotation_file}")
            return [], []

        num_samples = len(lines)
        with torch.no_grad(): # Disable gradient calculations
            for line in tqdm(lines, desc="Inferencing individual images"):
                try:
                    rel_path, label_str = line.strip().split()
                    label = int(label_str)
                    img_path = os.path.join(root_dir, rel_path)

                    if not os.path.exists(img_path):
                        print(f"❓ Warning: Image not found at {img_path}, skipping.")
                        continue

                    image = Image.open(img_path).convert('RGB')
                    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dim and move to device

                    outputs = model(input_tensor)
                    preds = outputs.argmax(dim=1)

                    y_pred.append(preds.item())
                    y_true.append(label)
                except Exception as e:
                    print(f"❓ Error processing line '{line.strip()}': {e}. Skipping.")

        print(f"✅ Completed single-image inference on {len(y_true)}/{num_samples} samples.")

    else:
        # --- Error Case: Insufficient Information ---
        print("❌ Error: Cannot perform inference. Provide either a DataLoader or (annotation_file, root_dir, and transform).")
        return [], []

    # --- Final Check ---
    if len(y_true) != len(y_pred):
         print(f"⚠️ Warning: Mismatch between number of true labels ({len(y_true)}) and predictions ({len(y_pred)}).")
    elif len(y_true) == 0:
         print("⚠️ Warning: No samples were processed during inference.")

    return y_true, y_pred

def report_classification(y_true, y_pred, class_names, save_path_base=None):
    """Generates and optionally saves classification report and confusion matrices."""
    if not y_true or not y_pred:
        print("⚠️ Cannot generate report: No true labels or predictions available.")
        return

    print("\n📊 Classification Report:")
    try:
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
        print(report)
    except ValueError as e:
        print(f"❌ Error generating classification report: {e}")
        print("   This might happen if predictions contain labels not present in true labels or vice-versa.")
        return # Stop if report fails

    print("\n📊 Confusion Matrix (Raw Counts):")
    try:
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))) # Ensure all classes are included
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(max(6, len(class_names)*0.8), max(6, len(class_names)*0.8)))
        disp.plot(cmap='Oranges', ax=ax, xticks_rotation=45, values_format='d')
        plt.title('Confusion Matrix (Raw Counts)')
        plt.tight_layout()
        if save_path_base:
            raw_cm_path = save_path_base + '_raw.png'
            plt.savefig(raw_cm_path, bbox_inches='tight')
            print(f"   Saved raw confusion matrix to: {raw_cm_path}")
        plt.show()
    except Exception as e:
        print(f"❌ Error plotting raw confusion matrix: {e}")


    print("\n📊 Normalized Confusion Matrix (by True Label):")
    try:
        # Calculate normalized CM, handle division by zero for classes with no true samples
        cm_normalized = confusion_matrix(y_true, y_pred, normalize='true', labels=np.arange(len(class_names)))
        # Replace NaNs resulting from division by zero with 0.0
        cm_normalized = np.nan_to_num(cm_normalized)

        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(max(6, len(class_names)*0.8), max(6, len(class_names)*0.8)))
        disp_norm.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='.2f') # Format as float
        plt.title('Normalized Confusion Matrix (by True Label)')
        plt.tight_layout()
        if save_path_base:
            norm_cm_path = save_path_base + '_normalized.png'
            plt.savefig(norm_cm_path, bbox_inches='tight')
            print(f"   Saved normalized confusion matrix to: {norm_cm_path}")
        plt.show()
    except Exception as e:
        print(f"❌ Error plotting normalized confusion matrix: {e}")
