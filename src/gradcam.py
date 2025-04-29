import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings
import os
from collections import defaultdict

# Helper function to get one image per class (assuming dataset yields (image, label))
def get_one_image_per_class(dataset):
    """Gets a dictionary mapping class index to a single image tensor for that class."""
    class_to_image = {}
    num_classes = len(dataset.classes)
    # Iterate through the dataset to find one image for each class
    # Use dataset.samples or dataset.imgs which store (path, label)
    # Then load the image using dataset.loader and apply transform
    image_paths_labels = dataset.imgs # Use the attribute storing (path, label) tuples

    # Keep track of which classes we've found
    found_classes = set()

    for img_path, label in image_paths_labels:
        if label not in found_classes:
            try:
                # Load and transform the image
                image = dataset.loader(img_path) # Use the dataset's loader
                if dataset.transform:
                    image = dataset.transform(image) # Apply the dataset's transform
                else:
                    # Apply a default minimal transform if none exists
                    warnings.warn("Dataset has no transform, applying default ToTensor for GradCAM.")
                    image = transforms.ToTensor()(image)

                # Ensure image is a tensor and add batch dimension
                if isinstance(image, torch.Tensor):
                    class_to_image[label] = image.unsqueeze(0)
                    found_classes.add(label)
                else:
                    warnings.warn(f"Loaded image for class {label} is not a tensor. Skipping.")

            except Exception as e:
                warnings.warn(f"Could not load/transform image {img_path} for class {label}: {e}")
                continue # Skip this image if loading/transform fails

        # Stop once we have one image for each class
        if len(found_classes) == num_classes:
            break

    if len(found_classes) < num_classes:
        warnings.warn(f"Could only find sample images for {len(found_classes)} out of {num_classes} classes.")

    return class_to_image


# --- Automatic Target Layer Finder ---
def find_target_layer(model):
    """Attempts to find a suitable final convolutional or attention block layer for GradCAM."""
    target_layer = None

    # 1. Check for common timm model structures (Vision Transformers, ConvNeXt, etc.)
    if hasattr(model, 'blocks') and isinstance(model.blocks, nn.Sequential) and len(model.blocks) > 0:
        # Try the norm layer of the last block (common in ViT-like models)
        last_block = model.blocks[-1]
        if hasattr(last_block, 'norm1'):
            target_layer = last_block.norm1
            print(f"🎯 Found target layer: Last block's norm1")
        elif hasattr(last_block, 'norm'): # Some blocks might just have 'norm'
             target_layer = last_block.norm
             print(f"🎯 Found target layer: Last block's norm")
        else: # Fallback to the last block itself if no norm layer found
            target_layer = last_block
            print(f"🎯 Found target layer: Last block of 'blocks'")

    # 2. Check for ResNet-like structures (layer4)
    elif hasattr(model, 'layer4') and isinstance(model.layer4, nn.Sequential) and len(model.layer4) > 0:
        # Use the last block/module within layer4
        target_layer = model.layer4[-1]
        print(f"🎯 Found target layer: Last module in 'layer4'")

    # 3. Check for DenseNet-like structures (features.denseblock)
    elif hasattr(model, 'features') and hasattr(model.features, 'denseblock4'): # Specific to DenseNet
        target_layer = model.features.denseblock4
        print(f"🎯 Found target layer: 'features.denseblock4'")
    elif hasattr(model, 'features') and isinstance(model.features, nn.Sequential) and len(model.features) > 0:
         # Use the last module in features (often a Conv layer or BatchNorm)
         target_layer = model.features[-1]
         print(f"🎯 Found target layer: Last module in 'features'")

    # 4. Check for EfficientNet/MobileNetV3 style 'conv_head' or final blocks
    elif hasattr(model, 'conv_head'):
        target_layer = model.conv_head
        print(f"🎯 Found target layer: 'conv_head'")
    elif hasattr(model, 'blocks') and isinstance(model.blocks, nn.ModuleList) and len(model.blocks) > 0:
         # For MobileNetV3/EfficientNetV2 where blocks are ModuleList
         target_layer = model.blocks[-1] # Use the last stage/block sequence
         print(f"🎯 Found target layer: Last item in 'blocks' (ModuleList)")


    # 5. Generic Fallback: Find the last Conv2d layer
    if target_layer is None:
        print("⏳ Target layer not found via common structures, searching for last Conv2d...")
        last_conv_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_name = name

        if last_conv_name:
            # Need to get the actual module object from the name
            try:
                target_layer = dict(model.named_modules())[last_conv_name]
                print(f"🎯 Found target layer: Last Conv2d layer named '{last_conv_name}'")
            except KeyError:
                 print(f"❌ Error accessing module '{last_conv_name}' by name.")
                 target_layer = None
        else:
            print("❌ Could not find any Conv2d layer as a fallback.")

    if target_layer is None:
        raise ValueError("❌ Could not automatically find a suitable target layer for GradCAM in this model architecture.")

    return target_layer

# --- GradCAM Setup and Computation ---
def setup_gradcam(model, target_layer=None):
    """
    Sets up hooks for GradCAM on the specified or automatically found target layer.

    Args:
        model (nn.Module): The PyTorch model (already on the correct device).
        target_layer (nn.Module, optional): The specific layer to hook. If None, attempts auto-detection.

    Returns:
        tuple: (model, gradcam_data, compute_gradcam_func)
    """
    model.eval() # Ensure model is in eval mode

    if target_layer is None:
        print("🤖 Attempting to automatically find target layer for GradCAM...")
        target_layer = find_target_layer(model)

    if target_layer is None:
         # This case should be handled by find_target_layer raising an error, but double-check
         raise RuntimeError("Failed to identify target layer for GradCAM.")

    print(f"✅ Registering GradCAM hooks on layer: {target_layer.__class__.__name__}")

    gradcam_data = {'features': None, 'gradients': None}

    # Hook functions
    def forward_hook(module, input, output):
        # Ensure we capture the output tensor correctly, handling tuples if necessary
        if isinstance(output, torch.Tensor):
            gradcam_data['features'] = output
        elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
             gradcam_data['features'] = output[0] # Take the first tensor if output is a sequence
             # warnings.warn(f"Target layer output is a sequence, using the first element of shape {output[0].shape} for GradCAM features.")
        else:
             warnings.warn(f"Unsupported output type from target layer: {type(output)}. GradCAM might fail.")
             gradcam_data['features'] = None


    def backward_hook(module, grad_input, grad_output):
        # grad_output is a tuple. We need the gradient w.r.t. the output of the layer.
        if isinstance(grad_output, tuple) and len(grad_output) > 0 and isinstance(grad_output[0], torch.Tensor):
            gradcam_data['gradients'] = grad_output[0]
        else:
            warnings.warn(f"Unsupported grad_output type in backward hook: {type(grad_output)}. GradCAM might fail.")
            gradcam_data['gradients'] = None

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Store handles to remove them later if needed
    gradcam_data['forward_handle'] = forward_handle
    gradcam_data['backward_handle'] = backward_handle


    def compute_gradcam(input_tensor, class_idx=None, target_size=None):
        """
        Computes the GradCAM heatmap for a given input and class index.

        Args:
            input_tensor (torch.Tensor): Input tensor (B, C, H, W), already on the correct device.
            class_idx (int, optional): Target class index. If None, uses the predicted class.
            target_size (tuple, optional): (height, width) to resize CAM to. If None, uses input tensor size.

        Returns:
            np.ndarray: GradCAM heatmap (H, W), normalized to [0, 1]. Returns None on error.
        """
        # Ensure hooks are still registered (they might be removed inadvertently)
        if not gradcam_data.get('forward_handle') or not gradcam_data.get('backward_handle'):
             print("❌ Error: GradCAM hooks seem to be missing. Cannot compute CAM.")
             return None

        # --- Forward pass ---
        output = model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item() # Get predicted class for the first item in batch

        # --- Backward pass ---
        model.zero_grad()
        try:
            # Create a one-hot vector for the target class score
            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1 # Focus on the first image in the batch for CAM
            # Propagate the gradient of the target class score
            output.backward(gradient=one_hot, retain_graph=True) # Retain graph if computing multiple CAMs
        except Exception as e:
            print(f"❌ Error during backward pass for GradCAM: {e}")
            return None


        # --- Get gradients and features ---
        gradients = gradcam_data['gradients']
        features = gradcam_data['features']

        if gradients is None or features is None:
            print("❌ Error: Failed to capture gradients or features from hooks.")
            return None

        # Ensure gradients and features are usable (handle potential issues from hooks)
        if not isinstance(gradients, torch.Tensor) or not isinstance(features, torch.Tensor):
             print("❌ Error: Captured gradients or features are not tensors.")
             return None
        if gradients.ndim < 3 or features.ndim < 3: # Expect at least (B, C, H, W) or (B, C, D)
             print(f"❌ Error: Unexpected dimensions for gradients ({gradients.ndim}) or features ({features.ndim}).")
             return None

        # --- Compute CAM ---
        # Use data for the first image in the batch (index 0)
        gradients = gradients[0] # Shape: (C, H, W) or (C, D)
        features = features[0]   # Shape: (C, H, W) or (C, D)

        # Global Average Pooling on gradients
        # Handle spatial (H, W) vs sequence (D) dimensions
        if gradients.ndim == 3: # Spatial (C, H, W)
            weights = torch.mean(gradients, dim=(1, 2)) # Shape: (C,)
        elif gradients.ndim == 2: # Sequence (C, D) - common in transformers
            weights = torch.mean(gradients, dim=1) # Shape: (C,)
        else:
            print(f"❌ Error: Unexpected gradient dimensions after batch selection: {gradients.ndim}")
            return None

        # Weighted sum of feature maps
        cam = torch.zeros(features.shape[1:], dtype=torch.float32, device=features.device) # Match feature map dims (H, W) or (D,)
        for i, w in enumerate(weights):
            if i < features.shape[0]: # Ensure index is valid
                 cam += w * features[i]
            else:
                 warnings.warn(f"Weight index {i} out of bounds for features shape {features.shape}. Skipping.")


        # Apply ReLU
        cam = F.relu(cam)

        # Normalize CAM to [0, 1]
        cam_min = torch.min(cam)
        cam_max = torch.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8) # Add epsilon for stability
        else:
            cam = torch.zeros_like(cam) # Avoid division by zero if CAM is flat

        # --- Resize CAM to target size ---
        cam = cam.detach().cpu().numpy() # Move to CPU and convert to numpy

        if target_size is None:
            target_size = (input_tensor.shape[2], input_tensor.shape[3]) # Use input H, W

        # Check if CAM is spatial before resizing
        if cam.ndim == 2: # Only resize if it's a 2D spatial map (H, W)
            try:
                cam = cv2.resize(cam, (target_size[1], target_size[0])) # cv2 uses (W, H)
            except Exception as resize_err:
                print(f"❌ Error resizing CAM: {resize_err}. Returning unresized CAM.")
        elif cam.ndim == 1:
             warnings.warn("CAM appears to be 1D (sequence-like), skipping spatial resize.")
             # Optionally, could try repeating or other visualization for 1D CAMs
        else:
             warnings.warn(f"CAM has unexpected dimensions ({cam.ndim}), skipping resize.")


        return cam

    # Return the model (in eval mode), the data dict (for potential later use/debugging), and the compute function
    return model, gradcam_data, compute_gradcam


def show_gradcam_on_image(image_tensor, cam, title="Grad-CAM", save_path=None, model_config=None):
    """
    Overlays the GradCAM heatmap on the original image and displays/saves it.

    Args:
        image_tensor (torch.Tensor): The original input image tensor (1, C, H, W) on CPU.
        cam (np.ndarray): The GradCAM heatmap (H, W), normalized [0, 1].
        title (str): Title for the plot.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
        model_config (dict, optional): Contains 'mean' and 'std' for denormalization.
                                       Uses ImageNet defaults if not provided.
    """
    if cam is None:
        print("⚠️ Cannot show GradCAM: CAM data is None.")
        return
    if image_tensor is None or image_tensor.numel() == 0:
        print("⚠️ Cannot show GradCAM: Image tensor is None or empty.")
        return

    # --- Denormalize Image ---
    try:
        image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() # (H, W, C)

        # Use mean/std from model_config if available, else default
        mean = np.array(model_config.get('mean', [0.485, 0.456, 0.406])) if model_config else np.array([0.485, 0.456, 0.406])
        std = np.array(model_config.get('std', [0.229, 0.224, 0.225])) if model_config else np.array([0.229, 0.224, 0.225])

        image = image * std + mean
        image = np.clip(image, 0, 1) # Clip to valid range [0, 1]
    except Exception as e:
        print(f"❌ Error denormalizing image: {e}. Displaying raw tensor if possible.")
        # Fallback: try to display without denormalization if it fails
        try:
            image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image = np.clip(image, 0, 1) # Still clip
        except Exception as fallback_e:
            print(f"❌ Error preparing image tensor for display: {fallback_e}")
            return # Cannot proceed

    # Ensure image is in uint8 format for OpenCV operations
    image_uint8 = np.uint8(image * 255)

    # --- Create Heatmap ---
    # Ensure CAM is 2D spatial before applying colormap
    if cam.ndim != 2:
        print(f"⚠️ CAM is not 2D (shape: {cam.shape}). Cannot generate heatmap overlay.")
        # Optionally display just the original image
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f"{title} (Original Image Only - CAM not 2D)")
        plt.axis("off")
        if save_path: plt.savefig(save_path.replace('.jpg', '_original.jpg')); plt.close()
        else: plt.show()
        return

    try:
        heatmap = np.uint8(255 * cam)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # Ensure heatmap_color has same dimensions as image_uint8 if possible
        if heatmap_color.shape[:2] != image_uint8.shape[:2]:
             heatmap_color = cv2.resize(heatmap_color, (image_uint8.shape[1], image_uint8.shape[0]))

        # Convert heatmap color from BGR (OpenCV default) to RGB (matplotlib default)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"❌ Error creating heatmap: {e}")
        return

    # --- Create Overlay ---
    try:
        overlay = cv2.addWeighted(image_uint8, 0.6, heatmap_color, 0.4, 0)
        # Convert overlay BGR to RGB
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"❌ Error creating overlay: {e}")
        overlay_rgb = image_uint8 # Fallback to showing original image if overlay fails

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14) # Add main title

    axes[0].imshow(image) # Show denormalized original image
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap_color_rgb) # Show RGB heatmap
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb) # Show RGB overlay
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # --- Save or Show ---
    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"   💾 Grad-CAM figure saved to: {save_path}")
            plt.close(fig) # Close the figure after saving
        except Exception as e:
            print(f"   ⚠️ Error saving Grad-CAM figure: {e}")
            plt.show() # Show if saving failed
    else:
        plt.show()


def generate_and_save_gradcam_per_class(model, dataset, save_dir="gradcam_results", model_config=None, device='cuda'):
    """
    Generates and saves GradCAM visualizations for one sample image per class.

    Args:
        model (nn.Module): The trained model (already on the correct device).
        dataset (Dataset): The dataset object (e.g., AnnotationDataset) to sample from.
                           Must have `classes`, `imgs`, `loader`, `transform` attributes.
        save_dir (str): Directory to save the output images.
        model_config (dict, optional): Model configuration containing mean/std for display.
        device (str or torch.device): Device to run the model on.
    """
    if not hasattr(dataset, 'classes') or not dataset.classes:
        print("❌ Error: Dataset object does not have a 'classes' attribute or it's empty.")
        return
    if not hasattr(dataset, 'imgs') or not hasattr(dataset, 'loader') or not hasattr(dataset, 'transform'):
         print("❌ Error: Dataset object missing required attributes: 'imgs', 'loader', or 'transform'.")
         return

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📸 Generating Grad-CAM visualizations in: {save_dir}")

    # Get sample images
    class_to_image = get_one_image_per_class(dataset)
    if not class_to_image:
        print("⚠️ No sample images found for GradCAM generation.")
        return

    # Setup GradCAM hooks
    try:
        model_cam, gradcam_data, compute_gradcam_func = setup_gradcam(model) # Use auto-detection
    except Exception as e:
        print(f"❌ Failed to setup GradCAM: {e}")
        return

    # Generate CAM for each class sample
    for class_idx, img_tensor_batch in class_to_image.items():
        if img_tensor_batch is None or img_tensor_batch.numel() == 0:
            print(f"❓ Skipping class {class_idx}: Invalid image tensor.")
            continue

        input_tensor = img_tensor_batch.to(device) # Move batch to device
        class_name = dataset.classes[class_idx]
        print(f"   Processing class: {class_idx} - {class_name}")

        try:
            # Compute GradCAM for the specific class index
            cam = compute_gradcam_func(input_tensor, class_idx=class_idx)

            if cam is not None:
                save_path = os.path.join(save_dir, f"gradcam_class_{class_idx}_{class_name}.png")
                # Pass the original image tensor (on CPU) and model_config for display
                show_gradcam_on_image(
                    img_tensor_batch.cpu(), # Ensure image is on CPU for plotting
                    cam,
                    title=f"Grad-CAM: Class '{class_name}' (idx {class_idx})",
                    save_path=save_path,
                    model_config=model_config # Pass config for denormalization
                )
            else:
                print(f"   ⚠️ Failed to compute GradCAM for class {class_name}.")

        except Exception as e:
            print(f"   ❌ Error generating GradCAM for class {class_name}: {e}")
            # Cleanup CUDA memory if an error occurs during loop
            if device.type == 'cuda': torch.cuda.empty_cache()


    # Optional: Remove hooks after generation if the model object won't be reused for CAM
    # try:
    #     if 'forward_handle' in gradcam_data: gradcam_data['forward_handle'].remove()
    #     if 'backward_handle' in gradcam_data: gradcam_data['backward_handle'].remove()
    #     print("   Hooks removed.")
    # except Exception as e:
    #     print(f"   ⚠️ Error removing hooks: {e}")

    print("✅ Grad-CAM generation complete.")
