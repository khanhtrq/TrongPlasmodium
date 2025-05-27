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


def setup_gradcam_pytorch(model, model_name):
    """Setup GradCAM using pytorch-grad-cam library."""
    print(f"🔥 Setting up pytorch-grad-cam for model: {model_name}")
    
    # Determine the target layer based on model architecture
    target_layers = []
    use_reshape_transform = False
    
    # Common patterns for different model architectures
    if 'deit' in model_name.lower() or 'vit' in model_name.lower():
        if hasattr(model, 'norm'):
            # Vision Transformer models with a norm layer
            target_layers = [model.norm]
            use_reshape_transform = True
        # Vision Transformer models
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            target_layers = [model.blocks[-1].norm1]  # Last transformer block
            use_reshape_transform = True
        elif hasattr(model, 'layers') and len(model.layers) > 0:
            target_layers = [model.layers[-1]]
            use_reshape_transform = True
    elif 'resnet' in model_name.lower():
        # ResNet models
        if hasattr(model, 'layer4'):
            target_layers = [model.layer4[-1]]
    elif 'efficientnet' in model_name.lower():
        # EfficientNet models
        if hasattr(model, 'features'):
            target_layers = [model.features[-1]]
    elif 'densenet' in model_name.lower():
        # DenseNet models
        if hasattr(model, 'features'):
            target_layers = [model.features.norm5]
    else:
        # Generic fallback - try to find the last convolutional layer
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                target_layers = [module]
    
    if not target_layers:
        print(f"⚠️ Could not automatically determine target layer for {model_name}")
        print("Available modules:")
        for name, module in model.named_modules():
            print(f"  {name}: {type(module).__name__}")
        raise ValueError("Could not find suitable target layer for GradCAM")
    
    print(f"   ✅ Using target layer: {target_layers}")
    
    # Initialize GradCAM with conditional reshape transform
    if use_reshape_transform:
        print(f"   🔄 Using reshape transform for transformer architecture")
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    else:
        cam = GradCAM(model=model, target_layers=target_layers)
    
    return cam


def compute_gradcam_pytorch(cam, input_tensor, target_class=None):
    """Compute GradCAM using pytorch-grad-cam."""
    try:
        # Prepare targets
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        # Return the first image's CAM (assuming batch size 1)
        return grayscale_cam[0]
    
    except Exception as e:
        print(f"❌ Error computing GradCAM: {e}")
        return None


def tensor_to_rgb_image(tensor, model_config=None):
    """Convert tensor to RGB image for visualization."""
    # Move to CPU and convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    image = tensor.cpu().numpy()
    
    # Transpose from CHW to HWC
    if image.shape[0] == 3:  # If channels first
        image = np.transpose(image, (1, 2, 0))
    
    # Denormalize if needed (common ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Apply denormalization
    image = image * std + mean
    
    # Clip to [0, 1] range
    image = np.clip(image, 0, 1)
    
    return image


def show_gradcam_pytorch(rgb_img, cam, title="GradCAM", save_path=None):
    """Show GradCAM visualization using pytorch-grad-cam utilities."""
    try:
        # Create the CAM visualization
        visualization = show_cam_on_image(rgb_img, cam, use_rgb=True)
        
        # Create figure
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
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")


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
        model.eval()
        print(f"   📍 Model moved to {device} and set to eval mode")
        
    except Exception as e:
        print(f"   ❌ Error loading checkpoint: {e}")
        raise e
    
    return model, input_size, transform, model_config


def predict_dataset(model, dataloader, device, class_names):
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


def analyze_predictions(predictions, true_labels, confidences, class_names, num_samples_per_class=1):
    """Analyze predictions to find correct and incorrect examples."""
    print(f"\n📈 Analyzing predictions...")
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)
    
    # Initialize storage
    correct_samples = defaultdict(list)  # class_idx -> [(sample_idx, confidence), ...]
    incorrect_samples_by_true_class = defaultdict(list)  # true_class -> [(sample_idx, pred_class, confidence), ...]
    
    # Analyze each prediction
    for idx, (pred, true, conf) in enumerate(zip(predictions, true_labels, confidences)):
        if pred == true:
            # Correct prediction
            correct_samples[true].append((idx, conf))
        else:
            # Incorrect prediction - organize by true class
            incorrect_samples_by_true_class[true].append((idx, pred, conf))
    
    # Sort correct samples by confidence (highest first) and take exactly 1 per class
    final_correct_samples = {}
    for class_idx in range(len(class_names)):
        if class_idx in correct_samples and correct_samples[class_idx]:
            # Sort by confidence and take the best one
            sorted_samples = sorted(correct_samples[class_idx], key=lambda x: x[1], reverse=True)
            final_correct_samples[class_idx] = [sorted_samples[0]]  # Take only the best one
            print(f"   ✅ Class {class_names[class_idx]}: Found 1 correct sample (conf: {sorted_samples[0][1]:.3f})")
        else:
            final_correct_samples[class_idx] = []
            print(f"   ⚠️ Class {class_names[class_idx]}: No correct predictions found!")
    
    # For incorrect samples, ensure we have at least one misclassification FROM each true class
    final_incorrect_samples = []
    
    print(f"\n   ❌ Analyzing incorrect predictions by true class:")
    for class_idx in range(len(class_names)):
        class_name = class_names[class_idx]
        if class_idx in incorrect_samples_by_true_class and incorrect_samples_by_true_class[class_idx]:
            # Sort by confidence (highest confidence mistakes first) and take top few
            sorted_mistakes = sorted(
                incorrect_samples_by_true_class[class_idx], 
                key=lambda x: x[2], 
                reverse=True
            )
            
            # Take top 2-3 mistakes for this true class to see variety of confusions
            num_mistakes_per_class = min(3, len(sorted_mistakes))
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
    print(f"      Correct samples: {sum(len(samples) for samples in final_correct_samples.values())} total")
    print(f"      Incorrect samples: {len(final_incorrect_samples)} total")
    print(f"      Classes with correct predictions: {sum(1 for samples in final_correct_samples.values() if samples)}/{len(class_names)}")
    print(f"      Classes with misclassifications: {len(set(x[1] for x in final_incorrect_samples))}/{len(class_names)}")
    
    return final_correct_samples, final_incorrect_samples


def generate_gradcam_analysis(model, dataset, correct_samples, incorrect_samples, 
                            class_names, model_config, device, save_dir):
    """Generate GradCAM visualizations using pytorch-grad-cam."""
    print(f"\n🔥 Generating GradCAM analysis with pytorch-grad-cam...")
    print(f"   Will generate exactly 1 correct image per class and comprehensive incorrect coverage")
    
    # Create directories
    correct_dir = os.path.join(save_dir, "correct_predictions")
    incorrect_dir = os.path.join(save_dir, "incorrect_predictions")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    
    # Setup GradCAM using pytorch-grad-cam
    try:
        # Get model name from model_config or try to infer
        model_name = getattr(model_config, 'name', 'unknown') if model_config else 'unknown'
        if model_name == 'unknown':
            model_name = model.__class__.__name__
        
        cam = setup_gradcam_pytorch(model, model_name)
        print(f"   ✅ pytorch-grad-cam setup successful")
    except Exception as e:
        print(f"❌ Failed to setup pytorch-grad-cam: {e}")
        return
    
    # Process correct predictions (exactly 1 per class)
    print(f"   📸 Processing correct predictions (1 per class)...")
    correct_count = 0
    for class_idx in range(len(class_names)):
        class_name = class_names[class_idx]
        samples = correct_samples.get(class_idx, [])
        
        if not samples:
            print(f"      ⚠️ Skipping {class_name}: No correct predictions available")
            continue
            
        sample_idx, confidence = samples[0]  # Take the single best sample
        
        try:
            # Get sample from dataset
            image, label = dataset[sample_idx]
            if image is None:
                print(f"      ⚠️ Skipping {class_name}: Image data is None")
                continue
            
            # Prepare input
            input_tensor = image.unsqueeze(0).to(device)
            
            # Compute GradCAM
            grayscale_cam = compute_gradcam_pytorch(cam, input_tensor, target_class=class_idx)
            
            if grayscale_cam is not None:
                # Convert tensor to RGB image
                rgb_img = tensor_to_rgb_image(image, model_config)
                
                save_path = os.path.join(
                    correct_dir, 
                    f"correct_{class_idx:02d}_{class_name}_conf{confidence:.3f}.png"
                )
                
                show_gradcam_pytorch(
                    rgb_img,
                    grayscale_cam,
                    title=f"✅ Correct: {class_name} (conf: {confidence:.3f})",
                    save_path=save_path
                )
                correct_count += 1
                print(f"      ✅ Generated for {class_name} (conf: {confidence:.3f})")
            else:
                print(f"      ❌ Failed to compute GradCAM for {class_name}")
                
        except Exception as e:
            print(f"      ❌ Error processing {class_name}: {e}")
            continue
    
    print(f"   📊 Generated {correct_count}/{len(class_names)} correct prediction visualizations")
    
    # Process incorrect predictions
    print(f"   📸 Processing incorrect predictions (one image per mistake)...")
    incorrect_count = 0
    
    # Group by true class for organized output
    incorrect_by_true_class = defaultdict(list)
    for sample_idx, true_class, pred_class, confidence in incorrect_samples:
        incorrect_by_true_class[true_class].append((sample_idx, pred_class, confidence))
    
    for true_class_idx in range(len(class_names)):
        true_class_name = class_names[true_class_idx]
        mistakes = incorrect_by_true_class.get(true_class_idx, [])
        
        if not mistakes:
            print(f"      ⚠️ No misclassifications found for true class: {true_class_name}")
            continue
            
        print(f"      Processing {len(mistakes)} mistakes for true class: {true_class_name}")
        
        for mistake_num, (sample_idx, pred_class, confidence) in enumerate(mistakes):
            try:
                pred_name = class_names[pred_class]
                
                # Get sample from dataset
                image, label = dataset[sample_idx]
                if image is None:
                    continue
                
                # Prepare input
                input_tensor = image.unsqueeze(0).to(device)
                
                # Generate GradCAM for the predicted class (what the model focused on)
                grayscale_cam = compute_gradcam_pytorch(cam, input_tensor, target_class=pred_class)
                
                if grayscale_cam is not None:
                    # Convert tensor to RGB image
                    rgb_img = tensor_to_rgb_image(image, model_config)
                    
                    save_path = os.path.join(
                        incorrect_dir,
                        f"wrong_{true_class_idx:02d}_{mistake_num+1:02d}_{true_class_name}_wrongly_as_{pred_name}_conf{confidence:.3f}.png"
                    )
                    
                    show_gradcam_pytorch(
                        rgb_img,
                        grayscale_cam,
                        title=f"❌ Wrong: {true_class_name} predicted as {pred_name} (conf: {confidence:.3f})",
                        save_path=save_path
                    )
                    
                    incorrect_count += 1
                    print(f"         ✅ Generated: {true_class_name} wrongly as {pred_name}")
                else:
                    print(f"         ❌ Failed to compute GradCAM for {true_class_name} wrongly as {pred_name}")
                
            except Exception as e:
                print(f"         ❌ Error processing mistake {mistake_num+1} for {true_class_name}: {e}")
                continue
    
    print(f"   📊 Generated {incorrect_count} incorrect prediction visualizations")
    print(f"   ✅ GradCAM analysis complete with pytorch-grad-cam!")
    print(f"      Correct predictions saved in: {correct_dir}")
    print(f"      Incorrect predictions saved in: {incorrect_dir}")
    
    # Clean up
    del cam


def create_summary_report(correct_samples, incorrect_samples, class_names, save_dir):
    """Create a text summary of the analysis."""
    report_path = os.path.join(save_dir, "analysis_summary.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GRADCAM INFERENCE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Correct predictions summary
        f.write("CORRECT PREDICTIONS SUMMARY (1 per class):\n")
        f.write("-" * 40 + "\n")
        for class_idx in range(len(class_names)):
            class_name = class_names[class_idx]
            samples = correct_samples.get(class_idx, [])
            if samples:
                conf = samples[0][1]
                f.write(f"{class_name}: 1 sample, confidence: {conf:.3f}\n")
            else:
                f.write(f"{class_name}: NO correct predictions found\n")
        
        f.write("\n")
        
        # Class coverage analysis
        classes_with_correct = sum(1 for samples in correct_samples.values() if samples)
        f.write(f"Classes with correct predictions: {classes_with_correct}/{len(class_names)}\n")
        
        # Incorrect predictions summary
        f.write("\nINCORRECT PREDICTIONS SUMMARY (comprehensive coverage):\n")
        f.write("-" * 50 + "\n")
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
    
    print(f"📄 Analysis summary saved to: {report_path}")


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
    
    # Extract parameters
    data_dir = config['data_dir']
    results_dir = config['results_dir']
    class_names = config.get('class_names', None)
    num_classes = len(class_names) if class_names else None
    
    # Model selection (modify these as needed)
    model_name = "deit3_small_patch16_224.fb_in22k_ft_in1k"  # Change this to your desired model
    model_checkpoint = r"X:\datn\7cls_results\deit3_small_patch16_224.fb_in22k_ft_in1k\PlasmodiumClassification-1\results_kaggle\deit3_small_patch16_224.fb_in22k_ft_in1k\deit3_small_patch16_224.fb_in22k_ft_in1k_best.pth"
    
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
                class_names=class_names
            )
    
    if test_dataset is None:
        print("❌ Could not load test dataset")
        return
    
    print(f"   ✅ Loaded test dataset with {len(test_dataset)} samples")
    
    # Load model
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
    
    # Perform predictions
    try:
        predictions, true_labels, confidences = predict_dataset(
            model, test_loader, device, class_names
        )
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print("💡 This might be a multiprocessing issue. Consider running with num_workers=0")
        return
    
    # Analyze predictions
    correct_samples, incorrect_samples = analyze_predictions(
        predictions, true_labels, confidences, class_names, num_samples_per_class=1
    )
    
    # Create output directory in the same location as the model checkpoint
    model_dir = os.path.dirname(model_checkpoint)  # Get the directory containing the .pth file
    analysis_dir = os.path.join(model_dir, "gradcam_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"\n📁 Analysis will be saved alongside model checkpoint:")
    print(f"   Model: {model_checkpoint}")
    print(f"   Analysis: {analysis_dir}")
    
    # Generate GradCAM analysis
    generate_gradcam_analysis(
        model, test_dataset, correct_samples, incorrect_samples,
        class_names, model_config, device, analysis_dir
    )
    
    # Create summary report
    create_summary_report(correct_samples, incorrect_samples, class_names, analysis_dir)
    
    print(f"\n🎉 Analysis complete! Results saved alongside the model:")
    print(f"📁 Analysis directory: {analysis_dir}")
    print(f"   - Correct predictions: {os.path.join(analysis_dir, 'correct_predictions')}")
    print(f"   - Incorrect predictions: {os.path.join(analysis_dir, 'incorrect_predictions')}")
    print(f"   - Summary report: {os.path.join(analysis_dir, 'analysis_summary.txt')}")


if __name__ == "__main__":
    main()
