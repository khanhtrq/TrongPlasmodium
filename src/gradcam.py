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

# pytorch-grad-cam imports
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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


def find_target_layers_pytorch_gradcam(model, model_name=None):
    """
    Find appropriate target layers for pytorch-grad-cam based on model architecture.
    Returns a list of target layers suitable for GradCAM.
    """
    print("🔍 Finding target layers for pytorch-grad-cam...")
    
    target_layers = []
    
    # Detect model type from name or architecture
    if model_name:
        model_type = model_name.lower()
    else:
        model_type = model.__class__.__name__.lower()
    
    print(f"   Detected model type: {model_type}")
    
    # Vision Transformer models (DeiT, ViT, etc.)
    if any(keyword in model_type for keyword in ['deit', 'vit', 'vision_transformer']):
        print("   🤖 Handling Vision Transformer architecture...")
        
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            # Use the last transformer block's layer norm
            last_block = model.blocks[-1]
            if hasattr(last_block, 'norm1'):
                target_layers = [last_block.norm1]
                print(f"   ✅ Selected last block norm1: {type(last_block.norm1).__name__}")
            elif hasattr(last_block, 'ln_1'):  # Some ViT variants
                target_layers = [last_block.ln_1]
                print(f"   ✅ Selected last block ln_1: {type(last_block.ln_1).__name__}")
            else:
                target_layers = [last_block]
                print(f"   ✅ Selected entire last block: {type(last_block).__name__}")
    
    # ResNet models
    elif 'resnet' in model_type:
        print("   🤖 Handling ResNet architecture...")
        if hasattr(model, 'layer4') and len(model.layer4) > 0:
            target_layers = [model.layer4[-1]]
            print(f"   ✅ Selected layer4[-1]: {type(model.layer4[-1]).__name__}")
    
    # EfficientNet models
    elif 'efficientnet' in model_type:
        print("   🤖 Handling EfficientNet architecture...")
        if hasattr(model, 'features'):
            target_layers = [model.features[-1]]
            print(f"   ✅ Selected features[-1]: {type(model.features[-1]).__name__}")
        elif hasattr(model, 'conv_head'):
            target_layers = [model.conv_head]
            print(f"   ✅ Selected conv_head: {type(model.conv_head).__name__}")
    
    # DenseNet models
    elif 'densenet' in model_type:
        print("   🤖 Handling DenseNet architecture...")
        if hasattr(model, 'features') and hasattr(model.features, 'norm5'):
            target_layers = [model.features.norm5]
            print(f"   ✅ Selected features.norm5: {type(model.features.norm5).__name__}")
    
    # Generic fallback
    if not target_layers:
        print("   🔍 Using generic fallback to find target layers...")
        
        # Look for the last convolutional or normalization layer
        last_conv = None
        last_norm = None
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                last_norm = module
        
        if last_conv:
            target_layers = [last_conv]
            print(f"   ✅ Fallback: Selected last Conv2d layer")
        elif last_norm:
            target_layers = [last_norm]
            print(f"   ✅ Fallback: Selected last normalization layer")
    
    if not target_layers:
        print("   ❌ Could not find suitable target layers!")
        print("   Available modules:")
        for name, module in model.named_modules():
            print(f"      {name}: {type(module).__name__}")
        raise ValueError("Could not find suitable target layers for pytorch-grad-cam")
    
    print(f"   ✅ Final target layers: {[type(layer).__name__ for layer in target_layers]}")
    return target_layers


def setup_gradcam(model, target_layers=None, cam_algorithm='gradcam'):
    """
    Setup pytorch-grad-cam with the specified model and target layers.
    
    Args:
        model: PyTorch model
        target_layers: List of target layers. If None, auto-detect.
        cam_algorithm: Type of CAM algorithm ('gradcam', 'gradcam++', 'scorecam', etc.)
    
    Returns:
        tuple: (model, cam_object, compute_function)
    """
    print(f"🔥 Setting up pytorch-grad-cam with algorithm: {cam_algorithm}")
    
    model.eval()
    
    if target_layers is None:
        target_layers = find_target_layers_pytorch_gradcam(model)
    
    # Select CAM algorithm
    cam_algorithms = {
        'gradcam': GradCAM,
        'gradcam++': GradCAMPlusPlus,
        'scorecam': ScoreCAM,
        'ablationcam': AblationCAM,
        'xgradcam': XGradCAM,
        'eigencam': EigenCAM,
        'fullgrad': FullGrad
    }
    
    if cam_algorithm not in cam_algorithms:
        print(f"   ⚠️ Unknown algorithm '{cam_algorithm}', using 'gradcam'")
        cam_algorithm = 'gradcam'
    
    CAMClass = cam_algorithms[cam_algorithm]
    
    # Initialize CAM
    try:
        cam = CAMClass(
            model=model,
            target_layers=target_layers,
            use_cuda=torch.cuda.is_available()
        )
        print(f"   ✅ Successfully initialized {cam_algorithm.upper()}")
    except Exception as e:
        print(f"   ❌ Failed to initialize {cam_algorithm}: {e}")
        raise
    
    def compute_gradcam_func(input_tensor, class_idx=None):
        """
        Compute GradCAM for the given input and class.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Target class index. If None, use predicted class.
        
        Returns:
            numpy array: GradCAM heatmap
        """
        try:
            # Prepare targets
            if class_idx is not None:
                targets = [ClassifierOutputTarget(class_idx)]
            else:
                targets = None
            
            # Generate CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            
            # Return the first image's CAM (batch size assumed to be 1)
            return grayscale_cam[0] if len(grayscale_cam) > 0 else None
            
        except Exception as e:
            print(f"   ❌ Error computing GradCAM: {e}")
            return None
    
    # Return model, cam object, and compute function
    gradcam_data = {'cam_object': cam, 'target_layers': target_layers}
    
    return model, gradcam_data, compute_gradcam_func


def tensor_to_rgb_image(tensor, model_config=None):
    """Convert tensor to RGB image for visualization with pytorch-grad-cam."""
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and transpose
    if tensor.shape[0] == 3:  # CHW format
        image = tensor.cpu().numpy().transpose(1, 2, 0)  # HWC format
    else:
        image = tensor.cpu().numpy()
    
    # Denormalize using ImageNet stats (default) or provided config
    if model_config and 'mean' in model_config and 'std' in model_config:
        mean = np.array(model_config['mean'])
        std = np.array(model_config['std'])
    else:
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    
    # Denormalize
    image = image * std + mean
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    return image


def show_gradcam_on_image(image_tensor, cam, title="GradCAM", save_path=None, model_config=None):
    """
    Show GradCAM visualization using pytorch-grad-cam utilities.
    
    Args:
        image_tensor: Original image tensor (1, C, H, W) or (C, H, W)
        cam: GradCAM heatmap from pytorch-grad-cam
        title: Plot title
        save_path: Path to save the figure
        model_config: Configuration for denormalization
    """
    if cam is None:
        print("⚠️ Cannot show GradCAM: CAM data is None.")
        return
    
    if image_tensor is None or image_tensor.numel() == 0:
        print("⚠️ Cannot show GradCAM: Image tensor is None or empty.")
        return
    
    try:
        # Convert tensor to RGB image
        rgb_img = tensor_to_rgb_image(image_tensor, model_config)
        
        # Create visualization using pytorch-grad-cam utility
        visualization = show_cam_on_image(rgb_img, cam, use_rgb=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(rgb_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title("GradCAM Heatmap")
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(visualization)
        axes[2].set_title("GradCAM Overlay")
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"   💾 GradCAM figure saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()
            
    except Exception as e:
        print(f"❌ Error creating GradCAM visualization: {e}")


def generate_and_save_gradcam_per_class(model, dataset, save_dir="gradcam_results", 
                                      model_config=None, device='cuda', cam_algorithm='gradcam'):
    """
    Generate and save GradCAM visualizations using pytorch-grad-cam.
    
    Args:
        model: Trained model
        dataset: Dataset with classes, imgs, loader, transform attributes
        save_dir: Directory to save results
        model_config: Model configuration for denormalization
        device: Device to run on
        cam_algorithm: CAM algorithm to use
    """
    if not hasattr(dataset, 'classes') or not dataset.classes:
        print("❌ Error: Dataset object does not have a 'classes' attribute or it's empty.")
        return
    
    if not all(hasattr(dataset, attr) for attr in ['imgs', 'loader', 'transform']):
        print("❌ Error: Dataset object missing required attributes: 'imgs', 'loader', or 'transform'.")
        return

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📸 Generating GradCAM visualizations with pytorch-grad-cam in: {save_dir}")

    # Get sample images
    class_to_image = get_one_image_per_class(dataset)
    if not class_to_image:
        print("⚠️ No sample images found for GradCAM generation.")
        return

    # Setup pytorch-grad-cam
    try:
        model_cam, gradcam_data, compute_gradcam_func = setup_gradcam(model, cam_algorithm=cam_algorithm)
    except Exception as e:
        print(f"❌ Failed to setup pytorch-grad-cam: {e}")
        return

    # Generate CAM for each class sample
    success_count = 0
    for class_idx, img_tensor_batch in class_to_image.items():
        if img_tensor_batch is None or img_tensor_batch.numel() == 0:
            print(f"❓ Skipping class {class_idx}: Invalid image tensor.")
            continue

        input_tensor = img_tensor_batch.to(device)
        class_name = dataset.classes[class_idx]
        print(f"   Processing class: {class_idx} - {class_name}")

        try:
            # Compute GradCAM
            cam_result = compute_gradcam_func(input_tensor, class_idx=class_idx)

            if cam_result is not None:
                save_path = os.path.join(save_dir, f"gradcam_class_{class_idx}_{class_name}.png")
                
                show_gradcam_on_image(
                    img_tensor_batch.cpu(),
                    cam_result,
                    title=f"GradCAM ({cam_algorithm.upper()}): Class '{class_name}' (idx {class_idx})",
                    save_path=save_path,
                    model_config=model_config
                )
                success_count += 1
            else:
                print(f"   ⚠️ Failed to compute GradCAM for class {class_name}.")

        except Exception as e:
            print(f"   ❌ Error generating GradCAM for class {class_name}: {e}")
            if device == 'cuda':
                torch.cuda.empty_cache()

    print(f"✅ GradCAM generation complete. Successfully processed {success_count}/{len(class_to_image)} classes.")
    
    # Clean up
    del gradcam_data['cam_object']
