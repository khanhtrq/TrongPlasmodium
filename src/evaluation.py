import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings

def infer_from_annotation(model, class_names, device, dataloader=None, annotation_file=None, root_dir=None, transform=None, save_txt=False, save_txt_path="inference_results.txt"):
    """
    Run inference on a dataset, prioritizing DataLoader for efficiency.

    Args:
        model: The trained model, already moved to the correct device and in eval mode.
        class_names (list): List of class names in the correct order.
        device: Device to run inference on ('cuda' or 'cpu').
        dataloader (DataLoader, optional): Preferred method. DataLoader for the dataset.
        annotation_file (str, optional): Fallback. Path to annotation file. Required if dataloader is None.
        root_dir (str, optional): Fallback. Root directory for images. Required if dataloader is None.
        transform (callable, optional): Fallback. Transform to apply. Required if dataloader is None.
        save_txt (bool, optional): If True, save inference results to a .txt file.
        save_txt_path (str, optional): Path to save the .txt file.

    Returns:
        tuple: (y_true, y_pred) - Lists of true labels and predicted labels. Returns ([], []) on error.
    """
    import torch.nn.functional as F

    model.eval() # Ensure model is in evaluation mode
    y_true = []
    y_pred = []
    num_classes = len(class_names)
    results_lines = []

    if dataloader is not None:
        # --- Preferred Method: Use DataLoader for Batch Inference ---
        print("\n🚀 Performing inference using DataLoader (efficient)...")
        dataset_size = len(dataloader.dataset)
        if dataset_size == 0:
            print("⚠️ Warning: DataLoader dataset is empty. Cannot perform inference.")
            return [], []

        processed_samples = 0
        with torch.no_grad(): # Disable gradient calculations for inference
            pbar = tqdm(dataloader, desc="Inferencing batches", total=len(dataloader), unit="batch")
            for batch_idx, (inputs, labels, *extras) in enumerate(pbar):
                # Support for datasets that return (inputs, labels, file_paths)
                file_paths = None
                if len(extras) > 0:
                    file_paths = extras[0]
                else:
                    # Try to get file_paths from dataset if possible
                    if hasattr(dataloader.dataset, 'samples'):
                        file_paths = [s[0] for s in dataloader.dataset.samples][batch_idx * inputs.size(0):(batch_idx + 1) * inputs.size(0)]
                    else:
                        file_paths = [f"sample_{processed_samples + i}" for i in range(inputs.size(0))]

                if inputs.numel() == 0 or labels.numel() == 0:
                    warnings.warn(f"Skipping empty batch during inference. Check data loading.")
                    continue

                inputs = inputs.to(device, non_blocking=True)
                labels_cpu = labels.cpu().numpy() # Keep labels on CPU

                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy() # Get predictions on CPU

                y_pred.extend(preds)
                y_true.extend(labels_cpu)
                processed_samples += len(labels_cpu)
                pbar.set_postfix({'processed': f'{processed_samples}/{dataset_size}'})

                if save_txt:
                    for i in range(len(labels_cpu)):
                        path = file_paths[i] if file_paths is not None else f"sample_{processed_samples - len(labels_cpu) + i}"
                        label = labels_cpu[i]
                        pred = preds[i]
                        scores = probs[i]
                        line = f"{path},{label},{pred}," + ",".join([f"{score:.6f}" for score in scores])
                        results_lines.append(line)

                del inputs, labels, outputs, preds, probs # Cleanup

        print(f"✅ Completed batch inference on {processed_samples}/{dataset_size} samples.")

    elif annotation_file and root_dir and transform:
        # --- Fallback Method: Load Images Individually ---
        warnings.warn("⚠️ DataLoader not provided. Falling back to slower single-image inference from annotation file. This is less efficient and error-prone.", UserWarning)
        print(f"   Annotation: {annotation_file}")
        print(f"   Root Dir: {root_dir}")

        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"❌ Error: Annotation file not found at {annotation_file}")
            return [], []

        num_samples = len(lines)
        processed_samples = 0
        with torch.no_grad(): # Disable gradient calculations
            pbar = tqdm(lines, desc="Inferencing individual images", total=num_samples, unit="image")
            for i, line in enumerate(pbar):
                parts = line.strip().split()
                if len(parts) != 2:
                    print(f"❓ Warning: Skipping malformed line {i+1}: '{line.strip()}'")
                    continue
                rel_path, label_str = parts
                try:
                    label = int(label_str)
                    if not (0 <= label < num_classes):
                        print(f"❓ Warning: Label {label} from file is out of range [0, {num_classes-1}]. Skipping line {i+1}.")
                        continue
                except ValueError:
                    print(f"❓ Warning: Non-integer label '{label_str}' on line {i+1}. Skipping.")
                    continue

                img_path = os.path.join(root_dir, rel_path)

                try:
                    if not os.path.exists(img_path):
                        continue

                    image = Image.open(img_path).convert('RGB')
                    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dim and move to device

                    outputs = model(input_tensor)
                    probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                    pred = outputs.argmax(dim=1).item() # Get single prediction

                    y_pred.append(pred)
                    y_true.append(label)
                    processed_samples += 1
                    pbar.set_postfix({'processed': f'{processed_samples}/{num_samples}'})

                    if save_txt:
                        line_txt = f"{rel_path},{label},{pred}," + ",".join([f"{score:.6f}" for score in probs])
                        results_lines.append(line_txt)

                except Exception as e:
                    print(f"❓ Error processing image '{img_path}' or running model: {e}. Skipping line {i+1}.")
                    if 'input_tensor' in locals(): del input_tensor
                    if 'outputs' in locals(): del outputs
                    if device.type == 'cuda': torch.cuda.empty_cache()

        print(f"✅ Completed single-image inference on {processed_samples}/{num_samples} samples.")

    else:
        # --- Error Case: Insufficient Information ---
        print("❌ Error: Cannot perform inference. Provide either a DataLoader or all of (annotation_file, root_dir, transform).")
        return [], []

    # --- Save results to txt if requested ---
    if save_txt and results_lines:
        try:
            with open(save_txt_path, "w", encoding="utf-8") as f:
                for line in results_lines:
                    f.write(line + "\n")
            print(f"💾 Inference results saved to: {save_txt_path}")
        except Exception as e:
            print(f"❌ Error saving inference results to txt: {e}")

    # --- Final Check ---
    if len(y_true) != len(y_pred):
        print(f"⚠️ Warning: Mismatch between number of true labels ({len(y_true)}) and predictions ({len(y_pred)}). Results might be unreliable.")
    elif not y_true: # Check if the list is empty
        print("⚠️ Warning: No samples were successfully processed during inference.")

    return y_true, y_pred

def report_classification(y_true, y_pred, class_names, save_path_base=None):
    """Generates, prints, and optionally saves classification report and confusion matrices."""
    if not y_true or not y_pred:
        print("⚠️ Cannot generate report: No true labels or predictions available.")
        return

    if len(y_true) != len(y_pred):
        print("⚠️ Warning: Length mismatch between y_true and y_pred. Report may be inaccurate.")
        # Optionally truncate to the shorter length? Or proceed with caution.
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    num_classes = len(class_names)
    target_labels = np.arange(num_classes) # Ensure labels 0..N-1 are considered

    print("\n" + "="*20 + " Classification Report " + "="*20)
    try:
        # Calculate overall accuracy separately for clarity
        overall_accuracy = accuracy_score(y_true, y_pred)
        print(f"📊 Overall Accuracy: {overall_accuracy:.4f}")

        # Generate detailed report
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            labels=target_labels, # Ensure all classes are in the report
            digits=4,
            zero_division=0 # Report 0 for metrics with zero denominators
        )
        print(report)

        # Save report to file if path provided
        if save_path_base:
            report_path = save_path_base + '_report.txt'
            try:
                with open(report_path, 'w') as f:
                    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
                    f.write(report)
                print(f"   💾 Classification report saved to: {report_path}")
            except Exception as e:
                print(f"   ⚠️ Error saving classification report: {e}")

    except ValueError as e:
        print(f"❌ Error generating classification report: {e}")
        print("   This might happen if predictions contain labels not present in true labels or vice-versa, even after attempting to align.")
        # Print unique values to help debug
        print(f"   Unique True Labels: {np.unique(y_true)}")
        print(f"   Unique Predicted Labels: {np.unique(y_pred)}")
        return # Stop if report fails

    # --- Confusion Matrix Plotting ---
    plot_size = max(6, num_classes * 0.8) # Adjust plot size based on number of classes

    # 1. Confusion Matrix (Raw Counts)
    print("\n" + "-"*15 + " Confusion Matrix (Raw Counts) " + "-"*15)
    try:
        cm_raw = confusion_matrix(y_true, y_pred, labels=target_labels) # Use target_labels
        disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=class_names)

        fig_raw, ax_raw = plt.subplots(figsize=(plot_size, plot_size))
        disp_raw.plot(cmap='Oranges', ax=ax_raw, xticks_rotation=45, values_format='d')
        ax_raw.set_title('Confusion Matrix (Raw Counts)')
        plt.tight_layout()

        if save_path_base:
            raw_cm_path = save_path_base + '_cm_raw.png'
            try:
                plt.savefig(raw_cm_path, bbox_inches='tight', dpi=150)
                print(f"   💾 Raw confusion matrix saved to: {raw_cm_path}")
            except Exception as e:
                print(f"   ⚠️ Error saving raw confusion matrix plot: {e}")
        plt.show() # Show plot after attempting to save

    except Exception as e:
        print(f"❌ Error plotting raw confusion matrix: {e}")
        if 'fig_raw' in locals(): plt.close(fig_raw) # Close figure if error occurred


    # 2. Normalized Confusion Matrix (by True Label - Recall)
    print("\n" + "-"*10 + " Normalized Confusion Matrix (by True Label / Recall) " + "-"*10)
    try:
        # Calculate normalized CM, handle division by zero for classes with no true samples
        cm_norm_true = confusion_matrix(y_true, y_pred, labels=target_labels, normalize='true')
        # Replace NaNs resulting from division by zero (no true samples) with 0.0
        cm_norm_true = np.nan_to_num(cm_norm_true)

        disp_norm_true = ConfusionMatrixDisplay(confusion_matrix=cm_norm_true, display_labels=class_names)
        fig_norm_true, ax_norm_true = plt.subplots(figsize=(plot_size, plot_size))
        disp_norm_true.plot(cmap='Blues', ax=ax_norm_true, xticks_rotation=45, values_format='.2f') # Format as float
        ax_norm_true.set_title('Normalized Confusion Matrix (by True Label / Recall)')
        plt.tight_layout()

        if save_path_base:
            norm_true_cm_path = save_path_base + '_cm_norm_true.png'
            try:
                plt.savefig(norm_true_cm_path, bbox_inches='tight', dpi=150)
                print(f"   💾 Normalized (True) confusion matrix saved to: {norm_true_cm_path}")
            except Exception as e:
                print(f"   ⚠️ Error saving normalized (True) confusion matrix plot: {e}")
        plt.show() # Show plot after attempting to save

    except Exception as e:
        print(f"❌ Error plotting normalized (True) confusion matrix: {e}")
        if 'fig_norm_true' in locals(): plt.close(fig_norm_true)

    # 3. Normalized Confusion Matrix (by Predicted Label - Precision) - Optional but informative
    print("\n" + "-"*10 + " Normalized Confusion Matrix (by Predicted Label / Precision) " + "-"*10)
    try:
        cm_norm_pred = confusion_matrix(y_true, y_pred, labels=target_labels, normalize='pred')
        cm_norm_pred = np.nan_to_num(cm_norm_pred) # Handle cases where a class was never predicted

        disp_norm_pred = ConfusionMatrixDisplay(confusion_matrix=cm_norm_pred, display_labels=class_names)
        fig_norm_pred, ax_norm_pred = plt.subplots(figsize=(plot_size, plot_size))
        disp_norm_pred.plot(cmap='Greens', ax=ax_norm_pred, xticks_rotation=45, values_format='.2f')
        ax_norm_pred.set_title('Normalized Confusion Matrix (by Predicted Label / Precision)')
        plt.tight_layout()

        if save_path_base:
            norm_pred_cm_path = save_path_base + '_cm_norm_pred.png'
            try:
                plt.savefig(norm_pred_cm_path, bbox_inches='tight', dpi=150)
                print(f"   💾 Normalized (Pred) confusion matrix saved to: {norm_pred_cm_path}")
            except Exception as e:
                print(f"   ⚠️ Error saving normalized (Pred) confusion matrix plot: {e}")
        plt.show()

    except Exception as e:
        print(f"❌ Error plotting normalized (Pred) confusion matrix: {e}")
        if 'fig_norm_pred' in locals(): plt.close(fig_norm_pred)

    print("\n" + "="*20 + " End Report " + "="*20)

