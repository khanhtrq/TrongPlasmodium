import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
from PIL import Image
import cv2
import traceback  # Thêm import traceback

# pytorch-grad-cam imports
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.model_initializer import initialize_model
from src.data_loader import AnnotationDataset, ImageFolderWrapper, collate_fn_skip_error
from src.device_handler import get_device


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def find_target_layer(model):
    """
    Find suitable target layer for GradCAM using comprehensive heuristics.
    Adapted from gradcam.py with enhanced detection.
    """
    print(f"🔍 Diagnosing model structure for GradCAM target layer selection...")
    
    all_modules = list(model.named_modules())
    print(f"   Found {len(all_modules)} total modules")
    
    # Print last few modules for debugging
    num_layers_to_print = min(len(all_modules), 8)
    print(f"   Last {num_layers_to_print} modules:")
    for i in range(len(all_modules) - num_layers_to_print, len(all_modules)):
        name, module = all_modules[i]
        print(f"     - {name}: {type(module).__name__}")
    
    target_layer = None
    
    # 1. Check for timm model structures with 'blocks'
    if hasattr(model, 'blocks') and isinstance(model.blocks, (torch.nn.Sequential, torch.nn.ModuleList)) and len(model.blocks) > 0:
        last_block_candidate = model.blocks[-1]
        if hasattr(last_block_candidate, 'norm1'):
            target_layer = last_block_candidate.norm1
            print(f"🎯 Selected last block's 'norm1': {type(target_layer).__name__}")
        elif hasattr(last_block_candidate, 'norm'):
            target_layer = last_block_candidate.norm
            print(f"🎯 Selected last block's 'norm': {type(target_layer).__name__}")
        else:
            target_layer = last_block_candidate
            print(f"🎯 Selected last block: {type(target_layer).__name__}")
    
    # 2. Check for ResNet-like 'layer4'
    elif hasattr(model, 'layer4') and isinstance(model.layer4, torch.nn.Sequential) and len(model.layer4) > 0:
        target_layer = model.layer4[-1]
        print(f"🎯 Selected last module in 'layer4': {type(target_layer).__name__}")
    
    # 3. Check for features
    elif hasattr(model, 'features'):
        if hasattr(model.features, 'norm5'):
            target_layer = model.features.norm5
            print(f"🎯 Selected 'features.norm5': {type(target_layer).__name__}")
        elif isinstance(model.features, torch.nn.Sequential) and len(model.features) > 0:
            candidate = model.features[-1]
            if not isinstance(candidate, (torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d, torch.nn.MaxPool2d)):
                target_layer = candidate
                print(f"🎯 Selected last module in 'features': {type(target_layer).__name__}")
    
    # 4. Check for conv_head
    if target_layer is None and hasattr(model, 'conv_head'):
        target_layer = model.conv_head
        print(f"🎯 Selected 'conv_head': {type(target_layer).__name__}")
    
    # 5. Fallback: Find last Conv2d layer
    if target_layer is None:
        print("   Searching for last Conv2d layer...")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                print(f"🎯 Selected last Conv2d '{name}': {type(target_layer).__name__}")
    
    if target_layer is None:
        raise ValueError("Could not find suitable target layer for GradCAM")
    
    return target_layer


def setup_gradcam_hooks(model, target_layer=None):
    """Setup GradCAM using hook-based approach from gradcam.py"""
    model.eval()  # Critical: ensure eval mode
    
    if target_layer is None:
        target_layer = find_target_layer(model)
    
    print(f"✅ Setting up GradCAM hooks on: {target_layer.__class__.__name__}")
    
    gradcam_data = {'features': None, 'gradients': None}
    
    def forward_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            gradcam_data['features'] = output
        elif isinstance(output, (list, tuple)) and len(output) > 0:
            gradcam_data['features'] = output[0]
    
    def backward_hook(module, grad_input, grad_output):
        if isinstance(grad_output, tuple) and len(grad_output) > 0:
            gradcam_data['gradients'] = grad_output[0]
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    gradcam_data['forward_handle'] = forward_handle
    gradcam_data['backward_handle'] = backward_handle
    
    return gradcam_data


def compute_gradcam_hooks(model, gradcam_data, input_tensor, target_class=None):
    """Compute GradCAM using hook-based approach from gradcam.py"""
    try:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        print(f"         🔍 Input shape: {input_tensor.shape}, Target class: {target_class}")
        
        # Ensure gradients are enabled
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        print(f"         📊 Model output shape: {output.shape}, max: {output.max().item():.3f}")
        
        # Backward pass
        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get captured data
        gradients = gradcam_data['gradients']
        features = gradcam_data['features']
        
        if gradients is None or features is None:
            print("         ❌ Failed to capture gradients or features from hooks")
            return None
        
        print(f"         📐 Features shape: {features.shape}, Gradients shape: {gradients.shape}")
        print(f"         📊 Gradients range: [{gradients.min().item():.6f}, {gradients.max().item():.6f}]")
        print(f"         📊 Features range: [{features.min().item():.6f}, {features.max().item():.6f}]")
        
        # Use first sample in batch
        gradients = gradients[0]  # (C, H, W)
        features = features[0]    # (C, H, W)
        
        # Global Average Pooling on gradients
        if gradients.ndim == 3:  # Spatial (C, H, W)
            weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        elif gradients.ndim == 2:  # Sequence (C, D)
            weights = torch.mean(gradients, dim=1)  # (C,)
        else:
            print(f"         ❌ Unexpected gradient dimensions: {gradients.ndim}")
            return None
        
        print(f"         🎯 Weights shape: {weights.shape}, range: [{weights.min().item():.6f}, {weights.max().item():.6f}]")
        
        # Weighted sum of feature maps
        cam = torch.zeros(features.shape[1:], dtype=torch.float32, device=features.device)
        for i, w in enumerate(weights):
            if i < features.shape[0]:
                cam += w * features[i]
        
        print(f"         📊 Raw CAM range: [{cam.min().item():.6f}, {cam.max().item():.6f}]")
        
        # Apply ReLU
        cam = F.relu(cam)
        
        print(f"         📊 After ReLU CAM range: [{cam.min().item():.6f}, {cam.max().item():.6f}]")
        
        # Check if CAM is all zeros after ReLU
        if torch.all(cam == 0):
            print("         ⚠️ WARNING: CAM is all zeros after ReLU! This suggests no positive activations.")
            return None
        
        # Normalize CAM to [0, 1]
        cam_min = torch.min(cam)
        cam_max = torch.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = torch.zeros_like(cam)
        
        print(f"         📊 Final normalized CAM range: [{cam.min().item():.6f}, {cam.max().item():.6f}]")
        
        # Convert to numpy and resize
        cam_np = cam.detach().cpu().numpy()
        
        if cam_np.ndim == 2:  # Spatial CAM
            target_size = (input_tensor.shape[2], input_tensor.shape[3])
            try:
                cam_resized = cv2.resize(cam_np, (target_size[1], target_size[0]))
                print(f"         ✅ CAM resized to {cam_resized.shape}")
                return cam_resized
            except Exception as e:
                print(f"         ❌ Error resizing CAM: {e}")
                return cam_np
        else:
            print(f"         ⚠️ CAM is not 2D (shape: {cam_np.shape}), returning as-is")
            return cam_np
            
    except Exception as e:
        print(f"         ❌ Error in GradCAM computation: {e}")
        traceback.print_exc()
        return None


def cleanup_gradcam_hooks(gradcam_data):
    """Clean up GradCAM hooks"""
    try:
        if 'forward_handle' in gradcam_data:
            gradcam_data['forward_handle'].remove()
        if 'backward_handle' in gradcam_data:
            gradcam_data['backward_handle'].remove()
        print("   🧹 GradCAM hooks cleaned up")
    except Exception as e:
        print(f"   ⚠️ Error cleaning up hooks: {e}")


def tensor_to_rgb_image(tensor, model_config=None):
    """Convert tensor to RGB image for visualization with proper denormalization."""
    # Move to CPU and convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    image = tensor.cpu().numpy()
    
    # Transpose from CHW to HWC
    if image.shape[0] == 3:  # If channels first
        image = np.transpose(image, (1, 2, 0))
    
    # Get normalization parameters from model_config if available
    if model_config and hasattr(model_config, 'normalization'):
        mean = np.array(model_config.normalization.get('mean', [0.485, 0.456, 0.406]))
        std = np.array(model_config.normalization.get('std', [0.229, 0.224, 0.225]))
    else:
        # Default ImageNet normalization - but check if image is already normalized
        # If the image values are roughly in [-2, 2] range, it's likely normalized
        if image.min() < -1.0 or image.max() > 1.5:
            # Image appears to be normalized, apply denormalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            print(f"         📊 Applying ImageNet denormalization (mean={mean}, std={std})")
        else:
            # Image appears to already be in [0,1] range
            mean = np.array([0.0, 0.0, 0.0])
            std = np.array([1.0, 1.0, 1.0])
            print(f"         📊 Image appears unnormalized, skipping denormalization")
    
    # Apply denormalization
    if not np.allclose(mean, 0) or not np.allclose(std, 1):
        image = image * std + mean
    
    # Clip to [0, 1] range
    image = np.clip(image, 0, 1)
    
    print(f"         📐 Final image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
    
    return image


def show_gradcam_pytorch(rgb_img, cam, title="GradCAM", save_path=None):
    """Show GradCAM visualization with enhanced display and better heatmap processing."""
    try:
        # Ensure RGB image is in proper range [0, 1]
        if rgb_img.max() > 1.0:
            rgb_img = rgb_img / 255.0
        rgb_img = np.clip(rgb_img, 0, 1)
        
        # Ensure cam is in proper range [0, 1] and properly normalized
        cam = np.clip(cam, 0, 1)
    
        # Apply slight gaussian smoothing to CAM for better visualization
        try:
            from scipy.ndimage import gaussian_filter
            cam_smooth = gaussian_filter(cam, sigma=1.0)
        except ImportError:
            print("         ⚠️ scipy not available, using original CAM without smoothing")
            cam_smooth = cam
        
        # Create the CAM visualization
        visualization = show_cam_on_image(rgb_img, cam_smooth, use_rgb=True, colormap=cv2.COLORMAP_JET)
        
        # Create figure with larger size for better visibility
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(rgb_img)
        axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap with better colormap and proper scaling
        im = axes[1].imshow(cam_smooth, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
        axes[1].set_title("GradCAM Heatmap", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Add colorbar for heatmap
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Activation Intensity', rotation=270, labelpad=15)
        
        # Overlay
        axes[2].imshow(visualization)
        axes[2].set_title("GradCAM Overlay", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"      💾 Saved: {os.path.basename(save_path)}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        traceback.print_exc()


def load_model_from_checkpoint(model_path, model_name, num_classes, device):
    """Load a trained model from checkpoint."""
    print(f"🔄 Loading model checkpoint from: {model_path}")
    
    # Initialize model architecture
    model, input_size, transform, model_config = initialize_model(
        model_name, 
        num_classes=num_classes, 
        use_pretrained=False,  # We'll load our trained weights
        feature_extract=False
    )
    
    # Load trained weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Loaded model state from checkpoint (with metadata)")
        else:
            model.load_state_dict(checkpoint)
            print(f"   ✅ Loaded model state directly")
        
        model.to(device)
        model.eval()  # Ensure model is in evaluation mode - CRITICAL!
        print(f"   📍 Model moved to {device} and set to eval mode")
        
    except Exception as e:
        print(f"   ❌ Error loading checkpoint: {e}")
        raise e
    
    return model, input_size, transform, model_config


def apply_class_remapping(config):
    """Apply class remapping configuration and return updated class names and mapping."""
    class_names = config.get('class_names', [])
    remapping_config = config.get('class_remapping', {})
    
    if not remapping_config.get('enabled', False):
        print("📋 Class remapping disabled - using original classes")
        return class_names, None, len(class_names)
    
    print("🔄 Applying class remapping...")
    mapping = remapping_config.get('mapping', {})
    final_class_names = remapping_config.get('final_class_names', [])
    
    # Convert string keys to int if needed
    mapping = {int(k): int(v) for k, v in mapping.items()}
    
    print(f"   Original classes: {len(class_names)}")
    print(f"   Mapping: {mapping}")
    print(f"   Final classes: {len(final_class_names)}")
    
    # Use final class names if provided, otherwise derive from original
    if final_class_names:
        remapped_class_names = final_class_names
    else:
        # Create remapped class names by keeping unique target classes
        remapped_class_names = []
        used_indices = set()
        for i in range(len(class_names)):
            target_idx = mapping.get(i, i)
            if target_idx not in used_indices:
                remapped_class_names.append(class_names[i])
                used_indices.add(target_idx)
    
    print(f"   ✅ Remapped to {len(remapped_class_names)} classes: {remapped_class_names}")
    
    return remapped_class_names, mapping, len(remapped_class_names)


def remap_labels_batch(labels, mapping):
    """Remap a batch of labels according to the mapping."""
    if mapping is None:
        return labels
    
    remapped = labels.clone()
    for original, target in mapping.items():
        remapped[labels == original] = target
    
    return remapped


def predict_dataset(model, dataloader, device, class_names, class_mapping=None):
    """Perform predictions on entire dataset and return detailed results."""
    print(f"\n🔮 Performing predictions on dataset...")
    
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    all_image_paths = []
    
    model.eval()
    total_samples = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if images is None or labels is None:  # Handle collate_fn_skip_error
                continue
                
            images = images.to(device)
            labels = labels.to(device)
            
            # Apply class remapping to labels if needed
            if class_mapping is not None:
                labels = remap_labels_batch(labels, class_mapping)
            
            # Forward pass
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
            # Track accuracy
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"   Processed {batch_idx * dataloader.batch_size} samples...")
    
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"   📊 Overall Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    return all_predictions, all_true_labels, all_confidences


def analyze_predictions_incorrect_only(predictions, true_labels, confidences, class_names):
    """Analyze predictions to find ONLY incorrect examples."""
    print(f"\n📈 Analyzing predictions (focusing on incorrect predictions only)...")
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)
    
    # Initialize storage for incorrect predictions only
    incorrect_samples_by_true_class = defaultdict(list)  # true_class -> [(sample_idx, pred_class, confidence), ...]
    
    # Count totals for statistics
    total_correct = 0
    total_incorrect = 0
    
    # Analyze each prediction
    for idx, (pred, true, conf) in enumerate(zip(predictions, true_labels, confidences)):
        if pred == true:
            total_correct += 1
        else:
            # Incorrect prediction - organize by true class
            total_incorrect += 1
            incorrect_samples_by_true_class[true].append((idx, pred, conf))
    
    print(f"   📊 Total predictions: {len(predictions)}")
    print(f"   ✅ Correct: {total_correct} ({total_correct/len(predictions)*100:.1f}%)")
    print(f"   ❌ Incorrect: {total_incorrect} ({total_incorrect/len(predictions)*100:.1f}%)")
    
    # For incorrect samples, ensure we have good coverage
    final_incorrect_samples = []
    
    print(f"\n   ❌ Analyzing incorrect predictions by true class:")
    for class_idx in range(len(class_names)):
        class_name = class_names[class_idx]
        if class_idx in incorrect_samples_by_true_class and incorrect_samples_by_true_class[class_idx]:
            # Sort by confidence (highest confidence mistakes first) and take multiple examples
            sorted_mistakes = sorted(
                incorrect_samples_by_true_class[class_idx], 
                key=lambda x: x[2], 
                reverse=True
            )
            
            # Take up to 5 mistakes per true class to get good coverage
            num_mistakes_per_class = min(5, len(sorted_mistakes))
            for i in range(num_mistakes_per_class):
                sample_idx, pred_class, conf = sorted_mistakes[i]
                final_incorrect_samples.append((sample_idx, class_idx, pred_class, conf))
            
            pred_classes_confused = [class_names[x[1]] for x in sorted_mistakes[:num_mistakes_per_class]]
            print(f"      {class_name}: {len(sorted_mistakes)} total mistakes, showing top {num_mistakes_per_class}")
            print(f"         Confused with: {', '.join(pred_classes_confused)}")
        else:
            print(f"      {class_name}: No misclassifications found (perfect accuracy for this class!)")
    
    # Sort final incorrect samples by confidence overall
    final_incorrect_samples = sorted(final_incorrect_samples, key=lambda x: x[3], reverse=True)
    
    # Print summary
    print(f"\n   📊 Final Analysis Summary:")
    print(f"      Will generate GradCAM for {len(final_incorrect_samples)} incorrect predictions")
    print(f"      Classes with misclassifications: {len(set(x[1] for x in final_incorrect_samples))}/{len(class_names)}")
    
    return final_incorrect_samples


def create_comprehensive_incorrect_summary(model, dataset, incorrect_samples, 
                                         class_names, model_config, device, save_dir):
    """Create a comprehensive grid summary of all incorrect predictions."""
    print(f"\n🎨 Creating comprehensive summary grid of incorrect predictions...")
    
    if not incorrect_samples:
        print("   ⚠️ No incorrect samples to summarize")
        return
    
    # Setup GradCAM
    try:
        gradcam_data = setup_gradcam_hooks(model)
    except Exception as e:
        print(f"❌ Failed to setup GradCAM for summary: {e}")
        return
    
    try:
        # Limit to a reasonable number for visualization (e.g., top 20 most confident mistakes)
        max_samples = min(20, len(incorrect_samples))
        top_incorrect = sorted(incorrect_samples, key=lambda x: x[3], reverse=True)[:max_samples]
        
        print(f"   📊 Creating summary for top {max_samples} most confident mistakes")
        
        # Calculate grid dimensions (try to make it roughly square)
        import math
        cols = math.ceil(math.sqrt(max_samples))
        rows = math.ceil(max_samples / cols)
        
        # Create a large figure with subplots
        # Each sample gets 3 columns (original, heatmap, overlay)
        fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 9, rows * 3))
        fig.suptitle(f"❌ Top {max_samples} Most Confident Incorrect Predictions", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Handle case where we only have one row or one sample
        if rows == 1 and cols == 1:
            axes = axes.reshape(1, -1)
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols * 3 == 1:
            axes = axes.reshape(-1, 1)
        
        sample_idx = 0
        
        for row in range(rows):
            for col in range(cols):
                if sample_idx >= len(top_incorrect):
                    # Hide unused subplots
                    for sub_col in range(3):
                        ax_idx = col * 3 + sub_col
                        if ax_idx < axes.shape[1]:
                            axes[row, ax_idx].axis('off')
                    continue
                
                # Get sample data
                dataset_idx, true_class, pred_class, confidence = top_incorrect[sample_idx]
                true_name = class_names[true_class]
                pred_name = class_names[pred_class]
                
                print(f"      Processing {sample_idx+1}/{max_samples}: {true_name} → {pred_name} (conf: {confidence:.3f})")
                
                try:
                    # Get image from dataset
                    image, label = dataset[dataset_idx]
                    if image is None:
                        sample_idx += 1
                        continue
                    
                    # Prepare input and compute GradCAM
                    input_tensor = image.unsqueeze(0).to(device)
                    cam = compute_gradcam_hooks(model, gradcam_data, input_tensor, target_class=pred_class)
                    
                    # Convert image to RGB
                    rgb_img = tensor_to_rgb_image(image, model_config)
                    
                    # Calculate subplot indices
                    original_ax = axes[row, col * 3] if rows > 1 else axes[col * 3]
                    heatmap_ax = axes[row, col * 3 + 1] if rows > 1 else axes[col * 3 + 1]
                    overlay_ax = axes[row, col * 3 + 2] if rows > 1 else axes[col * 3 + 2]
                    
                    # Original image
                    original_ax.imshow(rgb_img)
                    original_ax.set_title(f"#{sample_idx+1}\nTrue: {true_name}\nPred: {pred_name}\nConf: {confidence:.3f}", 
                                        fontsize=8, fontweight='bold')
                    original_ax.axis('off')
                    
                    if cam is not None:
                        # Normalize and smooth CAM
                        cam_normalized = np.clip(cam, 0, 1)
                        try:
                            from scipy.ndimage import gaussian_filter
                            cam_smooth = gaussian_filter(cam_normalized, sigma=0.8)
                        except ImportError:
                            cam_smooth = cam_normalized
                        
                        # Heatmap
                        im = heatmap_ax.imshow(cam_smooth, cmap='jet', vmin=0, vmax=1)
                        heatmap_ax.set_title("Heatmap", fontsize=8)
                        heatmap_ax.axis('off')
                        
                        # Overlay
                        try:
                            visualization = show_cam_on_image(rgb_img, cam_smooth, use_rgb=True, colormap=cv2.COLORMAP_JET)
                            overlay_ax.imshow(visualization)
                            overlay_ax.set_title("Overlay", fontsize=8)
                        except Exception as e:
                            print(f"         ⚠️ Error creating overlay for sample {sample_idx+1}: {e}")
                            overlay_ax.imshow(rgb_img)  # Fallback to original
                            overlay_ax.set_title("Overlay (Failed)", fontsize=8)
                        overlay_ax.axis('off')
                    else:
                        # If GradCAM failed, show placeholder
                        heatmap_ax.text(0.5, 0.5, "GradCAM\nFailed", ha='center', va='center', 
                                      transform=heatmap_ax.transAxes, fontsize=10)
                        heatmap_ax.axis('off')
                        
                        overlay_ax.imshow(rgb_img)
                        overlay_ax.set_title("No Overlay", fontsize=8)
                        overlay_ax.axis('off')
                
                except Exception as e:
                    print(f"         ❌ Error processing sample {sample_idx+1}: {e}")
                    # Show error placeholders
                    for sub_col, title in enumerate(["Original (Error)", "Heatmap (Error)", "Overlay (Error)"]):
                        ax_idx = col * 3 + sub_col
                        error_ax = axes[row, ax_idx] if rows > 1 else axes[ax_idx]
                        error_ax.text(0.5, 0.5, f"Error\n{title}", ha='center', va='center',
                                    transform=error_ax.transAxes, fontsize=8)
                        error_ax.axis('off')
                
                sample_idx += 1
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        summary_path = os.path.join(save_dir, "comprehensive_incorrect_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Comprehensive summary saved: {summary_path}")
        
        # Create a detailed text summary to accompany the image
        summary_text_path = os.path.join(save_dir, "comprehensive_summary_details.txt")
        with open(summary_text_path, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE INCORRECT PREDICTIONS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Top {max_samples} Most Confident Incorrect Predictions:\n\n")
            
            for i, (dataset_idx, true_class, pred_class, confidence) in enumerate(top_incorrect):
                true_name = class_names[true_class]
                pred_name = class_names[pred_class]
                f.write(f"{i+1:2d}. Sample #{dataset_idx}: {true_name} → {pred_name} (confidence: {confidence:.4f})\n")
            
            # Add confusion statistics
            f.write(f"\nCONFUSION PATTERNS:\n")
            f.write("-" * 20 + "\n")
            confusion_counts = defaultdict(int)
            for _, true_class, pred_class, _ in top_incorrect:
                true_name = class_names[true_class]
                pred_name = class_names[pred_class]
                confusion_counts[(true_name, pred_name)] += 1
            
            for (true_name, pred_name), count in sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{true_name} → {pred_name}: {count} times\n")
        
        print(f"   📄 Detailed summary text saved: {summary_text_path}")
    
    finally:
        # Clean up hooks
        cleanup_gradcam_hooks(gradcam_data)


def generate_gradcam_analysis_incorrect_only(model, dataset, incorrect_samples, 
                                           class_names, model_config, device, save_dir):
    """Generate GradCAM visualizations ONLY for incorrect predictions using hook-based approach."""
    print(f"\n🔥 Generating GradCAM analysis for INCORRECT predictions only...")
    print(f"   Will generate {len(incorrect_samples)} visualizations")
    
    # Create directory
    incorrect_dir = os.path.join(save_dir, "incorrect_predictions_gradcam")
    os.makedirs(incorrect_dir, exist_ok=True)
    
    # First, create the comprehensive summary
    create_comprehensive_incorrect_summary(model, dataset, incorrect_samples, 
                                         class_names, model_config, device, save_dir)
    
    # Then generate individual visualizations
    # Setup GradCAM using hook-based approach
    try:
        print(f"   🏗️ Setting up hook-based GradCAM for individual samples...")
        gradcam_data = setup_gradcam_hooks(model)
        print(f"   ✅ Hook-based GradCAM setup successful")
    except Exception as e:
        print(f"❌ Failed to setup hook-based GradCAM: {e}")
        traceback.print_exc()
        return
    
    # Process incorrect predictions
    print(f"   📸 Processing individual incorrect predictions...")
    incorrect_count = 0
    total_attempts = 0
    
    # Group by true class for organized output
    incorrect_by_true_class = defaultdict(list)
    for sample_idx, true_class, pred_class, confidence in incorrect_samples:
        incorrect_by_true_class[true_class].append((sample_idx, pred_class, confidence))
    
    try:
        for true_class_idx in range(len(class_names)):
            true_class_name = class_names[true_class_idx]
            mistakes = incorrect_by_true_class.get(true_class_idx, [])
            
            if not mistakes:
                continue
                
            print(f"      Processing {len(mistakes)} mistakes for true class: {true_class_name}")
            
            for mistake_num, (sample_idx, pred_class, confidence) in enumerate(mistakes):
                total_attempts += 1
                try:
                    pred_name = class_names[pred_class]
                    print(f"         🔄 Processing sample {sample_idx}: {true_class_name} → {pred_name}")
                    
                    # Get sample from dataset
                    image, label = dataset[sample_idx]
                    if image is None:
                        print(f"         ⚠️ Skipping: Image data is None")
                        continue
                    
                    # Prepare input
                    input_tensor = image.unsqueeze(0).to(device)
                    
                    # Generate GradCAM using hook-based approach
                    grayscale_cam = compute_gradcam_hooks(model, gradcam_data, input_tensor, target_class=pred_class)
                    
                    if grayscale_cam is not None:
                        # Convert tensor to RGB image
                        rgb_img = tensor_to_rgb_image(image, model_config)
                        
                        # Clean filename
                        clean_true_name = true_class_name.replace('/', '_').replace('\\', '_')
                        clean_pred_name = pred_name.replace('/', '_').replace('\\', '_')
                        
                        save_path = os.path.join(
                            incorrect_dir,
                            f"wrong_{true_class_idx:02d}_{mistake_num+1:02d}_{clean_true_name}_as_{clean_pred_name}_conf{confidence:.3f}.png"
                        )
                        
                        show_gradcam_pytorch(
                            rgb_img,
                            grayscale_cam,
                            title=f"❌ Wrong: {true_class_name} predicted as {pred_name} (conf: {confidence:.3f})",
                            save_path=save_path
                        )
                        
                        incorrect_count += 1
                        print(f"         ✅ Generated GradCAM: {true_class_name} wrongly as {pred_name}")
                    else:
                        print(f"         ❌ Failed to compute GradCAM for {true_class_name} wrongly as {pred_name}")
                    
                except Exception as e:
                    print(f"         ❌ Error processing mistake {mistake_num+1} for {true_class_name}: {e}")
                    traceback.print_exc()
                    continue
    
    finally:
        # Always clean up hooks
        cleanup_gradcam_hooks(gradcam_data)
    
    print(f"   📊 Generated {incorrect_count}/{total_attempts} individual GradCAM visualizations successfully")
    print(f"   ✅ Hook-based GradCAM analysis complete!")
    print(f"      Individual predictions saved in: {incorrect_dir}")


def create_summary_report_incorrect_only(incorrect_samples, class_names, save_dir):
    """Create a text summary focusing on incorrect predictions only."""
    report_path = os.path.join(save_dir, "incorrect_predictions_analysis.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GRADCAM INCORRECT PREDICTIONS ANALYSIS REPORT\n")
        f.write("=" * 55 + "\n\n")
        
        # Incorrect predictions summary
        f.write("INCORRECT PREDICTIONS SUMMARY:\n")
        f.write("-" * 35 + "\n")
        f.write(f"Total misclassifications analyzed: {len(incorrect_samples)}\n\n")
        
        # Group incorrect by true class
        incorrect_by_true = defaultdict(list)
        for sample_idx, true_cls, pred_cls, conf in incorrect_samples:
            incorrect_by_true[true_cls].append((pred_cls, conf))
        
        f.write("Misclassifications by true class:\n")
        for class_idx in range(len(class_names)):
            class_name = class_names[class_idx]
            mistakes = incorrect_by_true.get(class_idx, [])
            if mistakes:
                f.write(f"\n{class_name} ({len(mistakes)} mistakes):\n")
                for pred_cls, conf in mistakes:
                    pred_name = class_names[pred_cls]
                    f.write(f"  - Predicted as {pred_name} (confidence: {conf:.3f})\n")
            else:
                f.write(f"\n{class_name}: Perfect accuracy - no mistakes!\n")
        
        # Overall confusion analysis
        f.write("\nMOST COMMON CONFUSIONS:\n")
        f.write("-" * 25 + "\n")
        confusion_dict = defaultdict(int)
        for _, true_cls, pred_cls, _ in incorrect_samples:
            true_name = class_names[true_cls]
            pred_name = class_names[pred_cls]
            confusion_dict[(true_name, pred_name)] += 1
        
        for (true_name, pred_name), count in sorted(confusion_dict.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{true_name} predicted as {pred_name}: {count} times\n")
        
        # Statistics
        f.write(f"\nSTATISTICS:\n")
        f.write("-" * 12 + "\n")
        classes_with_errors = len(set(x[1] for x in incorrect_samples))
        f.write(f"Classes with misclassifications: {classes_with_errors}/{len(class_names)}\n")
        f.write(f"Average confidence on wrong predictions: {np.mean([x[3] for x in incorrect_samples]):.3f}\n")
        f.write(f"Highest confidence wrong prediction: {max([x[3] for x in incorrect_samples]):.3f}\n")
        f.write(f"Lowest confidence wrong prediction: {min([x[3] for x in incorrect_samples]):.3f}\n")
    
    print(f"📄 Incorrect predictions analysis saved to: {report_path}")


def main():
    """Main inference and analysis function."""
    # Configuration
    config_path = 'config_local.yaml'
    
    # Load configuration
    print(f"📖 Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    # Apply class remapping
    class_names, class_mapping, num_classes = apply_class_remapping(config)
    
    # Extract parameters
    data_dir = config['data_dir']
    results_dir = config['results_dir']
    
    # Model selection (modify these as needed)
    model_name = "mobilenetv4_hybrid_medium.ix_e550_r384_in1k"  # Change this to your desired model
    model_checkpoint = r"X:\datn\seed42\5cls mbnv4 aug\PlasmodiumClassification-1\results_kaggle\mobilenetv4_hybrid_medium.ix_e550_r384_in1k\mobilenetv4_hybrid_medium.ix_e550_r384_in1k_best.pth"
    
    # Check if model checkpoint exists
    if not os.path.exists(model_checkpoint):
        print(f"❌ Model checkpoint not found: {model_checkpoint}")
        print("Available models:")
        if os.path.exists(results_dir):
            for item in os.listdir(results_dir):
                model_dir = os.path.join(results_dir, item)
                if os.path.isdir(model_dir):
                    checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                    if checkpoints:
                        print(f"   {item}: {checkpoints}")
        return
    
    # Setup device
    device, _ = get_device(config.get('device', {}).get('use_cuda', True))
    print(f"🖥️ Using device: {device}")
    
    # Load test dataset
    print(f"\n📂 Loading test dataset...")
    datasets_config = config.get('datasets', [])
    if not datasets_config:
        print("❌ No datasets configuration found")
        return
    
    # Use the first dataset configuration for test data
    dataset_config = datasets_config[0]
    test_dataset = None
    
    if dataset_config.get('type', 'annotation').lower() == 'annotation':
        test_annotation = dataset_config.get('annotation_test')
        test_root = dataset_config.get('annotation_test_root', data_dir)
        
        if test_annotation:
            test_path = os.path.join(data_dir, test_annotation) if not os.path.isabs(test_annotation) else test_annotation
            test_dataset = AnnotationDataset(
                test_path, 
                test_root, 
                transform=None,  # Will be set after model loading
                class_names=config.get('class_names', [])  # Use original class names for dataset
            )
    
    if test_dataset is None:
        print("❌ Could not load test dataset")
        return
    
    print(f"   ✅ Loaded test dataset with {len(test_dataset)} samples")
    
    # Load model (use remapped num_classes)
    model, input_size, transform, model_config = load_model_from_checkpoint(
        model_checkpoint, model_name, num_classes, device
    )
    
    # Update dataset transform
    if transform:
        test_dataset.transform = transform
        print(f"   🔄 Applied model-specific transform to dataset")
    
    # Create dataloader with Windows-compatible settings
    print(f"   🔧 Creating DataLoader (Windows-optimized: num_workers=0)...")
    try:
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,  # Use smaller batch size for inference
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid Windows multiprocessing issues
            collate_fn=collate_fn_skip_error,
            pin_memory=False  # Disable pin_memory for CPU compatibility
        )
        print(f"   ✅ DataLoader created successfully")
    except Exception as e:
        print(f"   ❌ Error creating DataLoader: {e}")
        print(f"   💡 Tip: Try setting num_workers=0 in your config for Windows compatibility")
        return
    
    # Perform predictions with class remapping
    try:
        predictions, true_labels, confidences = predict_dataset(
            model, test_loader, device, class_names, class_mapping
        )
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print("💡 This might be a multiprocessing issue. Consider running with num_workers=0")
        return
    
    # Analyze predictions - ONLY incorrect ones
    incorrect_samples = analyze_predictions_incorrect_only(
        predictions, true_labels, confidences, class_names
    )
    
    # Create output directory in the same location as the model checkpoint
    model_dir = os.path.dirname(model_checkpoint)  # Get the directory containing the .pth file
    analysis_dir = os.path.join(model_dir, "gradcam_incorrect_only")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"\n📁 Analysis will be saved alongside model checkpoint:")
    print(f"   Model: {model_checkpoint}")
    print(f"   Analysis: {analysis_dir}")
    
    # Generate GradCAM analysis - ONLY for incorrect predictions
    generate_gradcam_analysis_incorrect_only(
        model, test_dataset, incorrect_samples,
        class_names, model_config, device, analysis_dir
    )
    
    # Create summary report for incorrect predictions only
    create_summary_report_incorrect_only(incorrect_samples, class_names, analysis_dir)
    
    print(f"\n🎉 Analysis complete! Results saved alongside the model:")
    print(f"📁 Analysis directory: {analysis_dir}")
    print(f"   - Incorrect predictions GradCAM: {os.path.join(analysis_dir, 'incorrect_predictions_gradcam')}")
    print(f"   - Summary report: {os.path.join(analysis_dir, 'incorrect_predictions_analysis.txt')}")


if __name__ == "__main__":
    main()
