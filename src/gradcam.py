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
            # For ViT, try multiple blocks for better results
            last_block = model.blocks[-1]
            second_last_block = model.blocks[-2] if len(model.blocks) > 1 else None
            
            # Option 1: Last block's norm layers
            target_candidates = []
            if hasattr(last_block, 'norm2'):
                target_candidates.append(last_block.norm2)
            if hasattr(last_block, 'norm1'):
                target_candidates.append(last_block.norm1)
            if second_last_block and hasattr(second_last_block, 'norm2'):
                target_candidates.append(second_last_block.norm2)
                
            if target_candidates:
                target_layers = target_candidates[:2]  # Use top 2 candidates
                print(f"   ✅ Selected ViT norm layers: {[type(layer).__name__ for layer in target_layers]}")
            else:
                target_layers = [last_block]
                print(f"   ✅ Selected entire last block: {type(last_block).__name__}")
    
    # ResNet models
    elif 'resnet' in model_type:
        print("   🤖 Handling ResNet architecture...")
        if hasattr(model, 'layer4') and len(model.layer4) > 0:
            # Try both last and second-to-last layers
            target_layers = [model.layer4[-1]]
            if len(model.layer4) > 1:
                target_layers.append(model.layer4[-2])
            print(f"   ✅ Selected ResNet layers: {[type(layer).__name__ for layer in target_layers]}")
    
    # EfficientNet models  
    elif 'efficientnet' in model_type:
        print("   🤖 Handling EfficientNet architecture...")
        if hasattr(model, 'features') and len(model.features) > 0:
            target_layers = [model.features[-1]]
            if len(model.features) > 1:
                target_layers.append(model.features[-2])
            print(f"   ✅ Selected EfficientNet features: {[type(layer).__name__ for layer in target_layers]}")
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
        print("   🔍 Using enhanced generic fallback to find target layers...")
        
        # Look for the last few layers of different types
        conv_layers = []
        norm_layers = []
        activation_layers = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                norm_layers.append(module)
            elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                activation_layers.append(module)
        
        # Prefer the last normalization layers, then conv layers
        if norm_layers:
            target_layers = norm_layers[-2:] if len(norm_layers) > 1 else [norm_layers[-1]]
            print(f"   ✅ Fallback: Selected last normalization layers")
        elif conv_layers:
            target_layers = conv_layers[-2:] if len(conv_layers) > 1 else [conv_layers[-1]]
            print(f"   ✅ Fallback: Selected last conv layers")
        elif activation_layers:
            target_layers = activation_layers[-2:] if len(activation_layers) > 1 else [activation_layers[-1]]
            print(f"   ✅ Fallback: Selected last activation layers")
    
    if not target_layers:
        print("   ❌ Could not find suitable target layers!")
        print("   Available modules:")
        for name, module in model.named_modules():
            print(f"      {name}: {type(module).__name__}")
        raise ValueError("Could not find suitable target layers for pytorch-grad-cam")
    
    print(f"   ✅ Final target layers: {[type(layer).__name__ for layer in target_layers]}")
    return target_layers


def setup_gradcam(model, target_layers=None, cam_algorithm='gradcam', debug_layers=False, sample_input=None):
    """
    Setup pytorch-grad-cam with the specified model and target layers.
    
    Args:
        model: PyTorch model
        target_layers: List of target layers. If None, auto-detect.
        cam_algorithm: Type of CAM algorithm ('gradcam', 'gradcam++', 'scorecam', etc.)
        debug_layers: Whether to run debug testing for target layers
        sample_input: Sample input for debugging (1, C, H, W)
    
    Returns:
        tuple: (model, cam_object, compute_function)
    """
    print(f"🔥 Setting up pytorch-grad-cam with algorithm: {cam_algorithm}")
    
    # CRITICAL: Set model to training mode for proper gradient computation
    model.train()
    
    # CRITICAL: Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad_(True)
    
    print(f"   ✅ Model set to training mode with gradients enabled")
    
    # Debug layers if requested
    if debug_layers and sample_input is not None:
        print("🐛 Running target layer debugging...")
        debug_results, recommended_layer = debug_target_layers(model, sample_input)
        if recommended_layer and target_layers is None:
            target_layers = [recommended_layer]
            print(f"   Using recommended layer from debug")
    
    if target_layers is None:
        target_layers = find_target_layers_pytorch_gradcam(model)
    
    print(f"   🎯 Final target layers: {[type(layer).__name__ for layer in target_layers]}")
    
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
            target_layers=target_layers
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
            # CRITICAL: Ensure model is in training mode for gradient computation
            model.train()
            
            # CRITICAL: Ensure input requires gradients
            input_tensor = input_tensor.detach().clone()
            input_tensor.requires_grad_(True)
            
            # Get model prediction first to understand what's happening
            with torch.no_grad():
                model.eval()  # Temporarily switch to eval for prediction
                outputs = model(input_tensor)
                predicted_class = outputs.argmax(1).item()
                confidence = torch.softmax(outputs, dim=1)[0, predicted_class].item()
                print(f"   🎯 Model prediction: class {predicted_class}, confidence: {confidence:.4f}")
                model.train()  # Switch back to training mode
            
            # If no class specified, use the predicted class
            if class_idx is None:
                class_idx = predicted_class
                print(f"   📌 Using predicted class {class_idx} for GradCAM")
            else:
                print(f"   📌 Using specified class {class_idx} for GradCAM")
            
            # Prepare targets
            targets = [ClassifierOutputTarget(class_idx)]
            
            # Generate CAM with detailed logging
            print(f"   🔥 Computing {cam_algorithm.upper()} for class {class_idx}...")
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            
            if len(grayscale_cam) > 0:
                cam_result = grayscale_cam[0]
                
                # Debug information about CAM values
                cam_min, cam_max = cam_result.min(), cam_result.max()
                cam_mean = cam_result.mean()
                cam_std = cam_result.std()
                
                print(f"   📊 CAM statistics:")
                print(f"      Min: {cam_min:.6f}, Max: {cam_max:.6f}")
                print(f"      Mean: {cam_mean:.6f}, Std: {cam_std:.6f}")
                print(f"      Shape: {cam_result.shape}")
                
                # Check if CAM is effectively empty
                if cam_max < 1e-6:
                    print(f"   ⚠️ WARNING: CAM values are extremely low (max: {cam_max:.6f})")
                    print(f"   💡 This might indicate wrong target layers or gradient issues")
                    
                    # Try alternative target layers
                    print(f"   🔄 Attempting alternative target layer detection...")
                    alternative_layers = find_alternative_target_layers(model)
                    if alternative_layers:
                        print(f"   🔄 Retrying with alternative layers: {[type(layer).__name__ for layer in alternative_layers]}")
                        # Recreate CAM with alternative layers
                        CAMClass = cam_algorithms[cam_algorithm]
                        alt_cam = CAMClass(model=model, target_layers=alternative_layers)
                        alt_grayscale_cam = alt_cam(input_tensor=input_tensor, targets=targets)
                        if len(alt_grayscale_cam) > 0:
                            alt_cam_result = alt_grayscale_cam[0]
                            alt_cam_max = alt_cam_result.max()
                            if alt_cam_max > cam_max:
                                print(f"   ✅ Alternative layers produced better result (max: {alt_cam_max:.6f})")
                                return alt_cam_result
                
                return cam_result
            else:
                print(f"   ❌ GradCAM returned empty result")
                return None
            
        except Exception as e:
            print(f"   ❌ Error computing GradCAM: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Return model, cam object, and compute function
    gradcam_data = {'cam_object': cam, 'target_layers': target_layers}
    
    return model, gradcam_data, compute_gradcam_func


def find_alternative_target_layers(model):
    """Find alternative target layers when the primary ones fail."""
    alternative_layers = []
    
    # Look for different types of layers that might work better
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
            alternative_layers.append(module)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            alternative_layers.append(module)
    
    # Return the last few activation layers
    return alternative_layers[-2:] if len(alternative_layers) > 1 else alternative_layers[-1:] if alternative_layers else []


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
        
        # Normalize CAM to [0, 1] range for better visualization
        cam_normalized = cam.copy()
        cam_min, cam_max = cam_normalized.min(), cam_normalized.max()
        
        print(f"   📊 Pre-normalization CAM range: [{cam_min:.6f}, {cam_max:.6f}]")
        
        if cam_max > cam_min:  # Avoid division by zero
            cam_normalized = (cam_normalized - cam_min) / (cam_max - cam_min)
        else:
            print(f"   ⚠️ WARNING: CAM has constant values ({cam_max:.6f}), creating uniform heatmap")
            cam_normalized = np.ones_like(cam_normalized) * 0.5
        
        print(f"   📊 Post-normalization CAM range: [{cam_normalized.min():.6f}, {cam_normalized.max():.6f}]")
        
        # Create visualization using pytorch-grad-cam utility
        visualization = show_cam_on_image(rgb_img, cam_normalized, use_rgb=True, colormap=cv2.COLORMAP_JET)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(rgb_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Heatmap with proper colormap
        im = axes[1].imshow(cam_normalized, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f"GradCAM Heatmap\n(min: {cam_min:.4f}, max: {cam_max:.4f})")
        axes[1].axis('off')
        # Add colorbar
        plt.colorbar(im, ax=axes[1], shrink=0.8)
        
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
        import traceback
        traceback.print_exc()


def generate_and_save_gradcam_per_class(model, dataset, save_dir="gradcam_results", 
                                      model_config=None, device='cuda', cam_algorithm='gradcam', debug_layers=False):
    """
    Generate and save GradCAM visualizations using pytorch-grad-cam.
    
    Args:
        model: Trained model
        dataset: Dataset with classes, imgs, loader, transform attributes
        save_dir: Directory to save results
        model_config: Model configuration for denormalization
        device: Device to run on
        cam_algorithm: CAM algorithm to use
        debug_layers: Whether to run debug testing for target layers
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

    # Get a sample for debugging if needed
    sample_input = None
    if debug_layers and class_to_image:
        first_class_idx = next(iter(class_to_image.keys()))
        sample_input = class_to_image[first_class_idx].to(device)

    # Setup pytorch-grad-cam
    try:
        model_cam, gradcam_data, compute_gradcam_func = setup_gradcam(
            model, 
            cam_algorithm=cam_algorithm,
            debug_layers=debug_layers,
            sample_input=sample_input
        )
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

def debug_target_layers(model, sample_input, class_idx=0):
    """
    Debug function to test different target layers and find the best one.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor (1, C, H, W)
        class_idx: Target class for testing
    
    Returns:
        dict: Results for different target layers
    """
    print("🔍 Debug: Testing different target layers...")
    
    # Get all potential layers
    potential_layers = []
    layer_names = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            potential_layers.append(module)
            layer_names.append(f"{name} ({type(module).__name__})")
    
    print(f"   Found {len(potential_layers)} potential target layers")
    
    results = {}
    
    # Test last few layers (most likely to work)
    test_layers = potential_layers[-min(5, len(potential_layers)):]
    test_names = layer_names[-min(5, len(layer_names)):]
    
    for i, (layer, name) in enumerate(zip(test_layers, test_names)):
        print(f"   Testing layer {i+1}/{len(test_layers)}: {name}")
        
        try:
            # Try to create GradCAM with this layer
            cam = GradCAM(model=model, target_layers=[layer])
            targets = [ClassifierOutputTarget(class_idx)]
            
            # Test computation
            grayscale_cam = cam(input_tensor=sample_input, targets=targets)
            
            if len(grayscale_cam) > 0:
                cam_result = grayscale_cam[0]
                cam_min, cam_max = cam_result.min(), cam_result.max()
                cam_mean = cam_result.mean()
                
                results[name] = {
                    'success': True,
                    'min': cam_min,
                    'max': cam_max,
                    'mean': cam_mean,
                    'layer': layer
                }
                
                print(f"      ✅ Success! CAM range: [{cam_min:.6f}, {cam_max:.6f}], mean: {cam_mean:.6f}")
            else:
                results[name] = {'success': False, 'error': 'Empty result'}
                print(f"      ❌ Failed: Empty result")
                
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
            print(f"      ❌ Failed: {e}")
    
    # Find best layer (highest max value that's not too extreme)
    best_layer = None
    best_score = -1
    
    for name, result in results.items():
        if result['success']:
            # Score based on max value (should be > 0 but not too high)
            score = result['max'] if 0.001 < result['max'] < 10 else 0
            if score > best_score:
                best_score = score
                best_layer = result['layer']
    
    if best_layer:
        print(f"   🏆 Recommended target layer: {best_layer}")
    else:
        print(f"   😞 No good target layer found")
    
    return results, best_layer
