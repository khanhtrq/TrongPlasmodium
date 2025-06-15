import os
import sys
import gc
import math
import random
import warnings
import traceback
import pprint
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision import transforms, datasets
import argparse

# Configure matplotlib for headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Local imports
from src.data_loader import AnnotationDataset, ImageFolderWrapper, CombinedDataset, collate_fn_skip_error, create_weighted_random_sampler, get_effective_sampler_config
from src.device_handler import get_device, setup_model_for_training
from src.model_initializer import initialize_model
from src.training import train_classifier_only  # Only import classifier training
from src.evaluation import infer_from_annotation, report_classification
from src.regularizers import MaxNorm_via_PGD, Normalizer
from src.gradcam import generate_and_save_gradcam_per_class
from src.loss import FocalLoss, F1Loss, get_criterion
from src.augment import (
    create_augmentation_strategy,
    get_timm_transform,
    MixupCutmixWrapper
)
from src.visualization import (
    plot_sample_images_per_class,
    plot_training_curves,
    plot_class_distribution_with_ratios,
    analyze_class_distribution_across_splits
)


def set_seed(seed):
    """
    Set seed for reproducibility across all random number generators.
    
    Args:
        seed (int): Random seed value
    """
    print(f"🎲 Setting seed to {seed} for reproducibility...")
    
    # Python random module
    random.seed(seed)
    
    # NumPy random module
    np.random.seed(seed)
    
    # PyTorch random module
    torch.manual_seed(seed)
    
    # PyTorch CUDA random module (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional settings for deterministic behavior on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ Seed {seed} has been set for all random number generators.")


def load_config(config_path='tuning_classifier_config.yaml'):
    """Loads YAML configuration file."""
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at {config_path}")
        exit()
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML configuration file: {e}")
        exit()
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading config: {e}")
        exit()


def init_regularizers(regularizer_config):
    """
    Initialize regularizers based on configuration.
    
    Args:
        regularizer_config (dict): Configuration dictionary for regularizers
        
    Returns:
        tuple: (max_norm_regularizer, tau_normalizer, tau_norm_frequency)
    """
    max_norm_regularizer = None
    tau_normalizer = None
    tau_norm_frequency = 1
    
    # Initialize Max-Norm regularizer
    max_norm_config = regularizer_config.get('max_norm', {})
    if max_norm_config.get('enabled', False):
        print(f"   🔧 Initializing Max-Norm regularizer...")
        thresh = max_norm_config.get('thresh', 1.0)
        lp_norm = max_norm_config.get('lp_norm', 2)
        tau = max_norm_config.get('tau', 1.0)
        
        max_norm_regularizer = MaxNorm_via_PGD(
            thresh=thresh,
            LpNorm=lp_norm,
            tau=tau
        )
        print(f"      ✅ Max-Norm regularizer created (thresh={thresh}, lp_norm={lp_norm}, tau={tau})")
    else:
        print(f"   📝 Max-Norm regularizer disabled")
    
    # Initialize Tau-Normalization regularizer
    tau_norm_config = regularizer_config.get('tau_normalization', {})
    if tau_norm_config.get('enabled', False):
        print(f"   🔧 Initializing Tau-Normalization regularizer...")
        lp_norm = tau_norm_config.get('lp_norm', 2)
        tau = tau_norm_config.get('tau', 1.0)
        tau_norm_frequency = tau_norm_config.get('apply_frequency', 1)
        
        tau_normalizer = Normalizer(
            LpNorm=lp_norm,
            tau=tau
        )
        print(f"      ✅ Tau-Normalization regularizer created (lp_norm={lp_norm}, tau={tau}, frequency={tau_norm_frequency})")
    else:
        print(f"   📝 Tau-Normalization regularizer disabled")
    
    return max_norm_regularizer, tau_normalizer, tau_norm_frequency


def load_datasets(datasets_config_list, data_dir, transform_train, transform_eval, class_remapping_config, final_class_names):
    """
    Load and prepare datasets for training.
    
    Returns:
        tuple: (train_datasets, val_datasets, test_datasets, first_dataset_loaded)
    """
    print("\n⏳ Loading datasets from configured sources...")
    train_datasets_list = []
    val_datasets_list = []
    test_datasets_list = []
    first_dataset_loaded = False

    for i, d_cfg in enumerate(datasets_config_list):
        print(f"   Loading Source {i+1} (Type: {d_cfg.get('type', 'N/A')})...")
        dataset_type = d_cfg.get('type', 'annotation').lower()
        current_train_dataset, current_val_dataset, current_test_dataset = None, None, None

        try:
            if dataset_type == 'annotation':
                ann_train_path = d_cfg.get('annotation_train')
                ann_val_path = d_cfg.get('annotation_val')
                ann_test_path = d_cfg.get('annotation_test')
                ann_root = d_cfg.get('annotation_root', data_dir)
                train_root = d_cfg.get('annotation_train_root', ann_root)
                val_root = d_cfg.get('annotation_val_root', ann_root)
                test_root = d_cfg.get('annotation_test_root', ann_root)

                def resolve_path(base, p):
                    return p if os.path.isabs(p) else os.path.join(base, p)

                if ann_train_path:
                    full_train_path = resolve_path(data_dir, ann_train_path)
                    print(f"      Train annotation: {full_train_path}")
                    current_train_dataset = AnnotationDataset(
                        annotation_file=full_train_path,
                        root_dir=train_root,
                        transform=transform_train,
                        class_remapping=class_remapping_config if class_remapping_config.get('enabled', False) else None
                    )
                    
                if ann_val_path:
                    full_val_path = resolve_path(data_dir, ann_val_path)
                    print(f"      Validation annotation: {full_val_path}")
                    current_val_dataset = AnnotationDataset(
                        annotation_file=full_val_path,
                        root_dir=val_root,
                        transform=transform_eval,
                        class_remapping=class_remapping_config if class_remapping_config.get('enabled', False) else None
                    )
                    
                if ann_test_path:
                    full_test_path = resolve_path(data_dir, ann_test_path)
                    print(f"      Test annotation: {full_test_path}")
                    current_test_dataset = AnnotationDataset(
                        annotation_file=full_test_path,
                        root_dir=test_root,
                        transform=transform_eval,
                        class_remapping=class_remapping_config if class_remapping_config.get('enabled', False) else None
                    )

            elif dataset_type == 'imagefolder':
                imgf_root = d_cfg.get('imagefolder_root')
                if not imgf_root:
                    raise ValueError("'imagefolder_root' is required for ImageFolder dataset type.")
                
                imgf_train_subdir = d_cfg.get('imagefolder_train_subdir')
                imgf_val_subdir = d_cfg.get('imagefolder_val_subdir')
                imgf_test_subdir = d_cfg.get('imagefolder_test_subdir')

                if imgf_train_subdir:
                    train_path = os.path.join(imgf_root, imgf_train_subdir)
                    print(f"      Train folder: {train_path}")
                    current_train_dataset = ImageFolderWrapper(
                        datasets.ImageFolder(root=train_path, transform=transform_train),
                        class_remapping=class_remapping_config if class_remapping_config.get('enabled', False) else None
                    )
                else:
                    current_train_dataset = ImageFolderWrapper(
                        datasets.ImageFolder(root=imgf_root, transform=transform_train),
                        class_remapping=class_remapping_config if class_remapping_config.get('enabled', False) else None
                    )

                if imgf_val_subdir:
                    val_path = os.path.join(imgf_root, imgf_val_subdir)
                    print(f"      Validation folder: {val_path}")
                    current_val_dataset = ImageFolderWrapper(
                        datasets.ImageFolder(root=val_path, transform=transform_eval),
                        class_remapping=class_remapping_config if class_remapping_config.get('enabled', False) else None
                    )

                if imgf_test_subdir:
                    test_path = os.path.join(imgf_root, imgf_test_subdir)
                    print(f"      Test folder: {test_path}")
                    current_test_dataset = ImageFolderWrapper(
                        datasets.ImageFolder(root=test_path, transform=transform_eval),
                        class_remapping=class_remapping_config if class_remapping_config.get('enabled', False) else None
                    )

            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")

            # Validate datasets and check class consistency
            dataset_to_check = current_train_dataset or current_val_dataset or current_test_dataset
            if dataset_to_check:
                if not first_dataset_loaded:
                    if hasattr(dataset_to_check, 'classes') and dataset_to_check.classes:
                        if final_class_names and final_class_names != dataset_to_check.classes:
                            warnings.warn(f"Class names mismatch between config and dataset {i+1}. Using dataset classes.")
                        final_class_names = dataset_to_check.classes
                        print(f"   First dataset loaded. Class names: {final_class_names}")
                        first_dataset_loaded = True
                    else:
                        warnings.warn(f"Dataset {i+1} has no 'classes' attribute or is empty.")
                else:
                    if hasattr(dataset_to_check, 'classes') and dataset_to_check.classes != final_class_names:
                        warnings.warn(f"Class name inconsistency in dataset {i+1}. Expected: {final_class_names}, Got: {dataset_to_check.classes}")
            else:
                warnings.warn(f"Dataset source {i+1} yielded no valid datasets (train/val/test all None).")

            # Add to lists if datasets were created
            if current_train_dataset: train_datasets_list.append(current_train_dataset)
            if current_val_dataset: val_datasets_list.append(current_val_dataset)
            if current_test_dataset: test_datasets_list.append(current_test_dataset)

        except Exception as e:
            print(f"❌ Error loading dataset source {i+1}: {e}")
            raise e

    return train_datasets_list, val_datasets_list, test_datasets_list, first_dataset_loaded


def display_confusion_matrix_stats(y_true, y_pred, class_names):
    """
    Display confusion matrix statistics in console for quick overview.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels  
        class_names (list): List of class names
    """
    try:
        from sklearn.metrics import confusion_matrix, accuracy_score
        
        print(f"\n📊 Quick Confusion Matrix Statistics:")
        print(f"   Total samples: {len(y_true)}")
        print(f"   Overall accuracy: {accuracy_score(y_true, y_pred):.4f}")
        
        # Calculate confusion matrix
        num_classes = len(class_names)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        
        print(f"\n   Per-class accuracy (recall):")
        for i, class_name in enumerate(class_names):
            if cm[i].sum() > 0:  # Avoid division by zero
                class_accuracy = cm[i, i] / cm[i].sum()
                total_samples = cm[i].sum()
                correct_samples = cm[i, i]
                print(f"      {class_name}: {class_accuracy:.4f} ({correct_samples}/{total_samples})")
            else:
                print(f"      {class_name}: N/A (no samples)")
        
        print(f"\n   Per-class precision:")
        for i, class_name in enumerate(class_names):
            if cm[:, i].sum() > 0:  # Avoid division by zero
                class_precision = cm[i, i] / cm[:, i].sum()
                total_predicted = cm[:, i].sum()
                correct_predicted = cm[i, i]
                print(f"      {class_name}: {class_precision:.4f} ({correct_predicted}/{total_predicted})")
            else:
                print(f"      {class_name}: N/A (no predictions)")
                
    except Exception as e:
        print(f"   ⚠️ Could not display confusion matrix stats: {e}")


def load_pretrained_model(model_path, model_name, num_classes, device, dropout_rate=0.0):
    """
    Load a pretrained model from checkpoint.
    
    Args:
        model_path (str): Path to the pretrained model checkpoint
        model_name (str): Name of the model architecture
        num_classes (int): Number of classes
        device (torch.device): Device to load the model on
        dropout_rate (float): Dropout rate for the model
        
    Returns:
        tuple: (model, model_config) - Loaded model and its configuration
    """
    print(f"📂 Loading pretrained model from: {model_path}")
    
    # Initialize model architecture
    model, input_size, model_specific_transform, model_config = initialize_model(
        model_name, 
        num_classes=num_classes, 
        use_pretrained=False,  # We'll load our own weights
        feature_extract=False, 
        drop_rate=dropout_rate
    )
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded model state dict from checkpoint")
                
                # Print additional checkpoint info if available
                if 'epoch' in checkpoint:
                    print(f"   Checkpoint epoch: {checkpoint['epoch']}")
                if 'best_val_acc' in checkpoint:
                    print(f"   Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
                if 'best_val_metric' in checkpoint:
                    print(f"   Best validation metric: {checkpoint['best_val_metric']:.4f}")
                    
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"✅ Loaded state dict from checkpoint")
            else:
                # Assume the entire dict is the state dict
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded model weights from checkpoint")
        else:
            # Assume it's a direct state dict
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights from checkpoint")
            
    except Exception as e:
        print(f"❌ Error loading pretrained model: {e}")
        print(f"   Falling back to using pretrained ImageNet weights...")
        # Reinitialize with pretrained weights as fallback
        model, input_size, model_specific_transform, model_config = initialize_model(
            model_name, 
            num_classes=num_classes, 
            use_pretrained=True, 
            feature_extract=False, 
            drop_rate=dropout_rate
        )
    
    return model, model_config


def find_pretrained_model_path(model_name, results_dir='./results'):
    """
    Automatically find pretrained model path for a given model name.
    
    Args:
        model_name (str): Name of the model
        results_dir (str): Base results directory
        
    Returns:
        str or None: Path to the pretrained model or None if not found
    """
    possible_paths = [
        os.path.join(results_dir, model_name, f'{model_name}_best.pth'),
        os.path.join(results_dir, model_name, 'model_best.pth'),
        os.path.join(results_dir, f'{model_name}_best.pth'),
        os.path.join(results_dir, model_name, 'best_model.pth'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"   🔍 Found pretrained model at: {path}")
            return path
    
    return None


def reinitialize_classifier(model, model_name="unknown"):
    """
    Reinitialize the classifier layer parameters.
    
    Args:
        model: The model whose classifier should be reinitialized
        model_name: Name of the model for logging purposes
    """
    print(f"🔄 Reinitializing classifier parameters for {model_name}...")
    
    classifier_reinitialized = False
    
    # Try to get classifier using get_classifier() method
    if hasattr(model, 'get_classifier') and callable(model.get_classifier):
        try:
            classifier_module = model.get_classifier()
            if isinstance(classifier_module, nn.Module):
                # Reinitialize weights and biases
                if hasattr(classifier_module, 'weight') and classifier_module.weight is not None:
                    nn.init.xavier_uniform_(classifier_module.weight)
                    print(f"   ✅ Reinitialized classifier weights using Xavier uniform")
                if hasattr(classifier_module, 'bias') and classifier_module.bias is not None:
                    nn.init.zeros_(classifier_module.bias)
                    print(f"   ✅ Reinitialized classifier bias to zeros")
                classifier_reinitialized = True
            else:
                print(f"   ⚠️ get_classifier() returned non-Module: {type(classifier_module)}")
        except Exception as e:
            warnings.warn(f"   ⚠️ Error calling model.get_classifier(): {e}")
    
    # Manual approach for common architectures if get_classifier failed
    if not classifier_reinitialized:
        print("   🔧 Trying manual classifier reinitialization...")
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
            if hasattr(model.classifier, 'weight') and model.classifier.weight is not None:
                nn.init.xavier_uniform_(model.classifier.weight)
                print(f"   ✅ Reinitialized model.classifier weights")
            if hasattr(model.classifier, 'bias') and model.classifier.bias is not None:
                nn.init.zeros_(model.classifier.bias)
                print(f"   ✅ Reinitialized model.classifier bias")
            classifier_reinitialized = True
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Module):
            if hasattr(model.fc, 'weight') and model.fc.weight is not None:
                nn.init.xavier_uniform_(model.fc.weight)
                print(f"   ✅ Reinitialized model.fc weights")
            if hasattr(model.fc, 'bias') and model.fc.bias is not None:
                nn.init.zeros_(model.fc.bias)
                print(f"   ✅ Reinitialized model.fc bias")
            classifier_reinitialized = True
        elif hasattr(model, 'head') and isinstance(model.head, nn.Module):
            if hasattr(model.head, 'weight') and model.head.weight is not None:
                nn.init.xavier_uniform_(model.head.weight)
                print(f"   ✅ Reinitialized model.head weights")
            if hasattr(model.head, 'bias') and model.head.bias is not None:
                nn.init.zeros_(model.head.bias)
                print(f"   ✅ Reinitialized model.head bias")
            classifier_reinitialized = True
    
    if not classifier_reinitialized:
        warnings.warn("   ⚠️ Could not find classifier layer to reinitialize")
    else:
        print(f"   🎯 Classifier reinitialization completed for {model_name}")


def main():
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description='Fine-tune classifier only')
    parser.add_argument('--re_init', action='store_true', 
                       help='Reinitialize classifier parameters before training', default=False)
    args = parser.parse_args()
    
    if args.re_init:
        print("🔄 Classifier reinitialization enabled via --re_init flag")
    
    # --- Configuration Loading ---
    config_file = 'tuning_classifier_config.yaml'  # Use the correct config file
    config = load_config(config_file)

    # --- Set Seed for Reproducibility ---
    seed = config.get('seed', 42)
    set_seed(seed)

    # --- Print Loaded Configuration for Debugging ---
    print("\n" + "="*20 + " Loaded Configuration " + "="*20)
    pprint.pprint(config, indent=2)
    print("="*60 + "\n")

    # --- Extract Configuration Parameters ---
    try:
        data_dir = config['data_dir']
        initial_batch_size = config['batch_size']
        num_workers = config['num_workers']        # Check for pretrained model configuration first
        pretrained_config = config.get('pretrained_model', {})
        if pretrained_config.get('enabled', False):
            print(f"🎯 Using pretrained model configuration from config:")
            model_name = pretrained_config.get('model_name')
            model_path = pretrained_config.get('model_path')
            
            if not model_name:
                print(f"❌ Error: 'model_name' must be specified in pretrained_model config")
                exit()
            
            # Auto-find model path if not provided
            if not model_path:
                print(f"   🔍 No model_path specified, searching for checkpoint...")
                model_path = find_pretrained_model_path(model_name, config.get('results_dir', './results'))
                if not model_path:
                    print(f"❌ Error: Could not find pretrained model for '{model_name}'")
                    print(f"   Please specify model_path in config or ensure model exists in results directory")
                    exit()
            
            print(f"   Model name: {model_name}")
            print(f"   Model path: {model_path}")
            
            # Validate model path exists
            if not os.path.exists(model_path):
                print(f"❌ Error: Model path not found: {model_path}")
                exit()
            
            model_names = [model_name]
            use_pretrained_checkpoint = True
            pretrained_model_path = model_path
        else:
            # Use standard model names from config (ImageNet pretrained weights)
            print(f"📋 Using model names from config file for classifier fine-tuning with ImageNet weights")
            model_names = config['model_names']
            use_pretrained_checkpoint = False
            pretrained_model_path = None
        
        datasets_config_list = config.get('datasets', [])
        if not datasets_config_list:
            raise ValueError("❌ 'datasets' list cannot be empty in config.")

        training_params = config.get('training', {})
        device_config = config.get('device', {})
        
        # Class Remapping Configuration
        class_remapping_config = config.get('class_remapping', {})
        if class_remapping_config.get('enabled', False):
            print(f"\n🔄 Class remapping is enabled: {class_remapping_config.get('mapping', {})}")

        # Classifier-only training parameters
        classifier_train_config = config.get('classifier_only_training', {})
        if not classifier_train_config.get('enabled', False):
            print("❌ Error: Classifier-only training must be enabled in config for this script.")
            print("   Please set 'classifier_only_training.enabled: true' in your config.yaml")
            exit()

        dropout_rate = float(training_params.get('dropout_rate', 0.0))
        use_amp = classifier_train_config.get('use_amp', training_params.get('use_amp', True))
        train_ratio = float(training_params.get('train_ratio', 1.0))
        if not (0.0 < train_ratio <= 1.0):
            warnings.warn(f"⚠️ Invalid train_ratio ({train_ratio}). Clamping to 1.0.")
            train_ratio = 1.0

        class_names = config.get('class_names', None)

    except KeyError as e:
        print(f"❌ Error: Missing key in configuration file: {e}")
        print("   Please ensure your config.yaml matches the expected structure.")
        exit()
    except (TypeError, ValueError) as e:
        print(f"❌ Error: Invalid value type or missing key in configuration file: {e}")
        print("   Please check the data types and structure in config.yaml.")
        exit()

    # --- Device Setup ---
    device, gpu_count = get_device(device_config.get('use_cuda', True), device_config.get('multi_gpu', True))
    print(f"\n🖥️ Device Selected: {device}")
    if device.type == 'cuda':
        print(f"   Number of GPUs available/requested: {gpu_count}")
        if gpu_count > 0:
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    # --- Class Information ---
    if class_remapping_config.get('enabled', False) and class_remapping_config.get('final_class_names'):
        final_class_names = class_remapping_config['final_class_names']
        print(f"\n📋 Using remapped class names: {final_class_names}")
    else:
        final_class_names = class_names
    
    num_classes = len(final_class_names) if final_class_names else None
    if final_class_names:
        print(f"\n📋 Class Information:")
        print(f"   Number of classes: {num_classes}")
        print(f"   Class names: {final_class_names}")
    else:
        print(f"\n📋 Class Information: Class names not provided, will be inferred from the first dataset.")

    # --- Training and Evaluation Loop ---
    for model_name in model_names:
        print(f"\n{'='*25} Fine-tuning Classifier: {model_name} {'='*25}")

        # --- Create Results Directory for this Model ---
        model_results_dir = os.path.join(config['results_dir'], f"{model_name}_classifier_finetune")
        os.makedirs(model_results_dir, exist_ok=True)
        print(f"   Results will be saved in: {model_results_dir}")

        # --- OOM Retry Loop ---
        current_batch_size = initial_batch_size
        model_trained_successfully = False
        history = None

        while current_batch_size >= 1:
            print(f"\n🔄 Attempting classifier fine-tuning for '{model_name}' with batch size: {current_batch_size}")
            try:                # --- Initialize Model (ImageNet pretrained or from checkpoint) ---
                temp_num_classes = num_classes if num_classes is not None else 5  # Use config class count or default
                
                if use_pretrained_checkpoint:
                    # Load from checkpoint
                    print(f"📂 Loading model '{model_name}' from checkpoint: {pretrained_model_path}")
                    model, model_config = load_pretrained_model(
                        pretrained_model_path, model_name, temp_num_classes, device, dropout_rate
                    )
                else:
                    # Load with ImageNet pretrained weights
                    print(f"📂 Initializing model '{model_name}' with ImageNet pretrained weights for classifier fine-tuning...")
                    model, input_size, model_specific_transform, model_config = initialize_model(
                        model_name, 
                        num_classes=temp_num_classes, 
                        use_pretrained=True,  # Use ImageNet pretrained weights
                        feature_extract=False, 
                        drop_rate=dropout_rate
                    )

                # --- Determine Transform to Use ---
                aug_config = config.get('augmentation', {})
                if aug_config.get('enabled', False):
                    print(f"   🎨 Using ADVANCED augmentation strategy: {aug_config.get('strategy', 'light')}")
                    aug_strategy = create_augmentation_strategy(config, model_config)
                    transform_train = aug_strategy.get_train_transform()
                    transform_eval = aug_strategy.get_eval_transform()
                else:
                    # Use basic transforms
                    input_size = model_config.get('input_size', 224)
                    transform_train = transforms.Compose([
                        transforms.Resize((input_size, input_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=model_config['mean'], std=model_config['std'])
                    ])
                    transform_eval = transforms.Compose([
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=model_config['mean'], std=model_config['std'])
                    ])

                print(f"\n🔧 Transform Configuration:")
                print(f"   Input size: {model_config.get('input_size', 224)}x{model_config.get('input_size', 224)}")
                print(f"   Augmentation enabled: {aug_config.get('enabled', False)}")
                print(f"   Train transform: {transform_train}")
                print(f"   Eval transform: {transform_eval}")
                # --- Load Datasets ---
                train_datasets_list, val_datasets_list, test_datasets_list, first_dataset_loaded = load_datasets(
                    datasets_config_list, data_dir, transform_train, transform_eval, 
                    class_remapping_config, final_class_names
                )

                if not train_datasets_list:
                    raise ValueError("❌ No training datasets were loaded successfully.")
                if not val_datasets_list:
                    warnings.warn("⚠️ No validation datasets were loaded. Validation phase will be skipped.")
                if not test_datasets_list:
                    warnings.warn("⚠️ No test datasets were loaded. Final evaluation will be skipped.")

                # Combine datasets if multiple
                if len(train_datasets_list) > 1:
                    print(f"   Combining {len(train_datasets_list)} training datasets...")
                    final_train_dataset_full = CombinedDataset(train_datasets_list)
                else:
                    final_train_dataset_full = train_datasets_list[0]

                # Apply train ratio if needed
                if train_ratio < 1.0:
                    subset_size = math.ceil(len(final_train_dataset_full) * train_ratio)
                    indices = list(range(subset_size))
                    final_train_dataset = Subset(final_train_dataset_full, indices)
                    if not hasattr(final_train_dataset, 'classes'): 
                        final_train_dataset.classes = final_class_names
                else:
                    final_train_dataset = final_train_dataset_full

                final_val_dataset = val_datasets_list[0] if val_datasets_list else None
                final_test_dataset = test_datasets_list[0] if test_datasets_list else None

                # Update num_classes if inferred from dataset
                if final_class_names is None and hasattr(final_train_dataset, 'classes'):
                    final_class_names = final_train_dataset.classes
                    num_classes = len(final_class_names)
                    print(f"   Inferred {num_classes} classes from dataset: {final_class_names}")

                # --- Setup Model for Training Device(s) ---
                model, device_actual = setup_model_for_training(
                    model,
                    use_cuda=device_config.get('use_cuda', True),
                    multi_gpu=device_config.get('multi_gpu', True)
                )
                print(f"   Model is now on device: {device_actual}")

                # --- Freeze All Parameters and Unfreeze Only Classifier ---
                print("\n🔒 Freezing all model parameters for classifier-only training...")
                for param in model.parameters():
                    param.requires_grad = False

                # Reinitialize classifier if requested
                if args.re_init:
                    reinitialize_classifier(model, model_name)

                # Get and unfreeze classifier module
                classifier_module = None
                if hasattr(model, 'get_classifier') and callable(model.get_classifier):
                    try:
                        classifier_module = model.get_classifier()
                        if isinstance(classifier_module, nn.Module):
                            for param in classifier_module.parameters():
                                param.requires_grad = True
                            print(f"   ✅ Unfroze classifier module: {type(classifier_module).__name__}")
                        else:
                            print(f"   ⚠️ get_classifier() returned non-Module: {type(classifier_module)}")
                    except Exception as e:
                        warnings.warn(f"   ⚠️ Error calling model.get_classifier(): {e}")
                else:
                    # Manual approach for common architectures
                    print("   ⚠️ model.get_classifier() method not found. Trying manual layer unfreezing...")
                    if hasattr(model, 'classifier'):
                        for param in model.classifier.parameters():
                            param.requires_grad = True
                        print(f"   ✅ Unfroze model.classifier")
                    elif hasattr(model, 'fc'):
                        for param in model.fc.parameters():
                            param.requires_grad = True
                        print(f"   ✅ Unfroze model.fc")
                    elif hasattr(model, 'head'):
                        for param in model.head.parameters():
                            param.requires_grad = True
                        print(f"   ✅ Unfroze model.head")
                    else:
                        warnings.warn("   ⚠️ Could not find classifier layer to unfreeze")

                # Collect trainable parameters
                classifier_params = [p for p in model.parameters() if p.requires_grad]
                if not classifier_params:
                    print(f"❌ No trainable parameters found for classifier-only training. Skipping model '{model_name}'.")
                    break

                num_trainable = sum(p.numel() for p in classifier_params)
                print(f"   Number of trainable parameters: {num_trainable}")

                # --- Create DataLoaders ---
                # Create validation and test dataloaders if datasets exist
                val_loader = None
                test_loader = None
                
                if final_val_dataset:
                    val_loader = DataLoader(
                        final_val_dataset,
                        batch_size=current_batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        collate_fn=collate_fn_skip_error,
                        persistent_workers=num_workers > 0
                    )
                    
                if final_test_dataset:
                    test_loader = DataLoader(
                        final_test_dataset,
                        batch_size=current_batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        collate_fn=collate_fn_skip_error,
                        persistent_workers=num_workers > 0
                    )

                # --- Create WeightedRandomSampler for classifier training if enabled ---
                main_sampler_config = config.get('weighted_random_sampler', {})
                cls_sampler_config = classifier_train_config.get('weighted_random_sampler', {})
                effective_sampler_config = get_effective_sampler_config(main_sampler_config, cls_sampler_config)

                train_sampler = None
                dataloaders = {'train': None}
                if val_loader: 
                    dataloaders['val'] = val_loader

                if effective_sampler_config.get('enabled', False):
                    try:
                        train_sampler = create_weighted_random_sampler(
                            final_train_dataset,
                            num_classes,
                            effective_sampler_config
                        )
                        print(f"   ✅ WeightedRandomSampler created for classifier training")
                        print(f"      Strategy: {effective_sampler_config.get('weight_calculation', 'inverse')}")

                        # Create new DataLoader for classifier training with sampler
                        train_loader = DataLoader(
                            final_train_dataset,
                            batch_size=current_batch_size,
                            sampler=train_sampler,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn_skip_error,
                            persistent_workers=num_workers > 0,
                            drop_last=True
                        )
                        dataloaders['train'] = train_loader

                    except Exception as e:
                        print(f"   ⚠️ Failed to create WeightedRandomSampler: {e}")
                        print(f"   📝 Falling back to standard random sampling")
                        train_sampler = None
                        train_loader = DataLoader(
                            final_train_dataset,
                            batch_size=current_batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn_skip_error,
                            persistent_workers=num_workers > 0,
                            drop_last=True
                        )
                        dataloaders['train'] = train_loader
                else:
                    print(f"   📝 WeightedRandomSampler disabled")
                    train_loader = DataLoader(
                        final_train_dataset,
                        batch_size=current_batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        collate_fn=collate_fn_skip_error,
                        persistent_workers=num_workers > 0,
                        drop_last=True
                    )
                    dataloaders['train'] = train_loader

                sampler_info = f"WeightedRandomSampler ({effective_sampler_config.get('weight_calculation', 'N/A')})" if train_sampler else "Random Shuffle"
                print(f"\n📦 DataLoaders created (Batch size: {current_batch_size}, Sampling: {sampler_info})")

                # --- Setup Optimizer for Classifier Training ---
                cls_optimizer_config = classifier_train_config.get('optimizer', {'type': 'Adam', 'params': {'lr': 1e-4}})
                cls_optimizer_type = cls_optimizer_config.get('type', 'Adam').lower()
                cls_optimizer_params = cls_optimizer_config.get('params', {})

                # Handle legacy lr configuration (lr might be directly in optimizer config)
                if 'lr' in cls_optimizer_config and 'lr' not in cls_optimizer_params:
                    cls_optimizer_params['lr'] = cls_optimizer_config['lr']

                if cls_optimizer_type == 'adam':
                    optimizer = optim.Adam(classifier_params, **cls_optimizer_params)
                elif cls_optimizer_type == 'adamw':
                    optimizer = optim.AdamW(classifier_params, **cls_optimizer_params)
                elif cls_optimizer_type == 'sgd':
                    optimizer = optim.SGD(classifier_params, **cls_optimizer_params)
                else:
                    warnings.warn(f"Unsupported optimizer: {cls_optimizer_type}. Using Adam.")
                    default_lr = cls_optimizer_params.get('lr', 1e-4)
                    optimizer = optim.Adam(classifier_params, lr=default_lr)

                print(f"\n🔧 Classifier Optimizer: {type(optimizer).__name__}")
                print(f"   Learning rate: {cls_optimizer_params.get('lr', 'default')}")
                print(f"   Weight decay: {cls_optimizer_params.get('weight_decay', 'default')}")

                # --- Setup Scheduler ---
                cls_scheduler_config = classifier_train_config.get('scheduler', {'type': 'ReduceLROnPlateau'})
                cls_scheduler_type = cls_scheduler_config.get('type', 'ReduceLROnPlateau').lower()
                
                if cls_scheduler_type == 'cosineannealinglr':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=classifier_train_config.get('num_epochs', 10),
                        eta_min=cls_scheduler_config.get('min_lr', 0)
                    )
                elif cls_scheduler_type == 'reducelronplateau':
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode=cls_scheduler_config.get('mode', 'max'),
                        factor=cls_scheduler_config.get('factor', 0.1),
                        patience=cls_scheduler_config.get('patience', 3),
                        verbose=True
                    )
                elif cls_scheduler_type == 'steplr':
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=cls_scheduler_config.get('step_size', 5),
                        gamma=cls_scheduler_config.get('gamma', 0.1)
                    )
                else:
                    scheduler = None

                # --- Setup Criterion ---
                # Calculate class distribution for loss functions that need it
                if hasattr(final_train_dataset_full, 'targets') and hasattr(final_train_dataset_full, 'classes'):
                    targets_np = np.array(final_train_dataset_full.targets)
                    cls_num_list = [int(np.sum(targets_np == i)) for i in range(len(final_train_dataset_full.classes))]
                else:
                    cls_num_list = [1] * num_classes

                # Load criterion configuration
                cls_criterion_a_name = classifier_train_config.get('criterion_a', 
                                      classifier_train_config.get('criterion', 'CrossEntropyLoss')).lower()
                cls_criterion_a_params = classifier_train_config.get('criterion_a_params', 
                                        classifier_train_config.get('criterion_params', {}))
                
                cls_criterion_b_name = classifier_train_config.get('criterion_b', '').lower()
                cls_criterion_b_params = classifier_train_config.get('criterion_b_params', {})
                cls_first_stage_epochs = classifier_train_config.get('first_stage_epochs', 0)

                # Add class information to criterion params if needed
                if cls_criterion_a_name == 'ldamloss':
                    cls_criterion_a_params['cls_num_list'] = cls_num_list
                if cls_criterion_b_name == 'ldamloss':
                    cls_criterion_b_params['cls_num_list'] = cls_num_list

                if cls_criterion_a_name in ['cbloss', 'classbalancedloss']:
                    cls_criterion_a_params['samples_per_cls'] = cls_num_list
                    cls_criterion_a_params['num_classes'] = num_classes
                if cls_criterion_b_name in ['cbloss', 'classbalancedloss']:
                    cls_criterion_b_params['samples_per_cls'] = cls_num_list
                    cls_criterion_b_params['num_classes'] = num_classes

                # Initialize criterions
                criterion_a = get_criterion(cls_criterion_a_name, num_classes, device, cls_criterion_a_params)
                criterion_b = None
                if cls_first_stage_epochs > 0 and cls_criterion_b_name:
                    criterion_b = get_criterion(cls_criterion_b_name, num_classes, device, cls_criterion_b_params)

                # Add noise_sigma parameters to optimizer if needed
                if cls_criterion_a_name in ['bmcloss', 'gailoss']:
                    optimizer.add_param_group({'params': criterion_a.noise_sigma, 'lr': 1e-2, 'name': 'noise_sigma_a'})
                if criterion_b and cls_criterion_b_name in ['bmcloss', 'gailoss']:
                    optimizer.add_param_group({'params': criterion_b.noise_sigma, 'lr': 1e-2, 'name': 'noise_sigma_b'})

                print(f"\n📉 Classifier Criterion: {cls_criterion_a_name}")
                if criterion_b:
                    print(f"   Dual criterion training: {cls_first_stage_epochs} epochs with {cls_criterion_a_name}, then {cls_criterion_b_name}")

                # --- Initialize Regularizers ---
                print(f"\n🔧 Initializing regularizers for classifier training...")
                cls_regularizer_config = classifier_train_config.get('regularization', {})
                max_norm_reg, tau_norm_reg, tau_freq = init_regularizers(cls_regularizer_config)

                if max_norm_reg:
                    optimizer.add_param_group({'params': max_norm_reg.tau, 'lr': 1e-2, 'name': 'max_norm_regularizer'})
                
                # --- Training Configuration ---
                num_epochs = classifier_train_config.get('num_epochs', 10)
                patience = classifier_train_config.get('patience', 3)
                clip_grad_norm = classifier_train_config.get('clip_grad_norm', 1.0)
                cls_train_ratio = classifier_train_config.get('train_ratio', train_ratio)

                model_save_path = os.path.join(model_results_dir, f'{model_name}_classifier_best.pth')
                log_save_path = os.path.join(model_results_dir, f'{model_name}_classifier_training_log.csv')

                print(f"\n🚀 Starting classifier-only fine-tuning for '{model_name}'...")
                print(f"   Epochs: {num_epochs}, Patience: {patience}")
                print(f"   Batch size: {current_batch_size}, Train ratio: {cls_train_ratio}")

                # --- Start Training ---
                model, history = train_classifier_only(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion_a,
                    criterion_b=criterion_b,
                    first_stage_epochs=cls_first_stage_epochs,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    num_epochs=num_epochs,
                    patience=patience,
                    use_amp=use_amp,
                    save_path=model_save_path,
                    log_path=log_save_path,
                    clip_grad_norm=clip_grad_norm,
                    train_ratio=cls_train_ratio,
                    init_best_val_metric=0.0,
                    max_norm_regularizer=max_norm_reg,
                    tau_normalizer=tau_norm_reg,
                    tau_norm_frequency=tau_freq,
                    mixup_fn=None  # Typically disabled for classifier-only training
                )

                model_trained_successfully = True
                print(f"✅ Classifier fine-tuning completed successfully for '{model_name}' with batch size {current_batch_size}.")
                break

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"\n🔥 CUDA Out of Memory detected for '{model_name}' with batch size {current_batch_size}!")
                    
                    # Cleanup
                    if 'model' in locals(): del model
                    if 'train_loader' in locals(): del train_loader
                    if 'optimizer' in locals(): del optimizer
                    if 'criterion_a' in locals(): del criterion_a
                    if 'val_loader' in locals() and val_loader: del val_loader
                    if 'test_loader' in locals() and test_loader: del test_loader
                    if 'scheduler' in locals() and scheduler: del scheduler
                    if 'criterion_b' in locals() and criterion_b: del criterion_b
                    if 'history' in locals() and history: del history
                    gc.collect()
                    
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                    # Reduce batch size
                    new_batch_size = current_batch_size // 2
                    print(f"   Reducing batch size from {current_batch_size} to {new_batch_size}.")
                    current_batch_size = new_batch_size

                    if current_batch_size < 1:
                        print(f"❌ Batch size reduced below 1. Cannot train. Skipping model '{model_name}'.")
                        break
                else:
                    print(f"❌ Unexpected RuntimeError during training: {e}")
                    traceback.print_exc()
                    break
            except Exception as e:
                print(f"❌ Unexpected error during training: {e}")
                traceback.print_exc()
                break

        # --- Post Training Processing ---
        if model_trained_successfully:
            # Plot training curves
            plot_file_path = os.path.join(model_results_dir, f'{model_name}_classifier_training_curves.png')
            if history:
                try:
                    plot_training_curves(
                        history, 
                        title_suffix=f"(Classifier-Only {model_name}, BS={current_batch_size})", 
                        save_path=plot_file_path
                    )
                except Exception as e:
                    print(f"⚠️ Could not plot training curves: {e}")

            # --- Model Evaluation ---
            if test_loader:
                print(f"\n🧪 Evaluating fine-tuned classifier: {model_name}")
                model.eval()
                report_base_path = os.path.join(model_results_dir, f'{model_name}_classifier_test_eval')
                
                try:
                    y_true, y_pred = infer_from_annotation(
                        model=model,
                        class_names=final_class_names,
                        device=device,
                        dataloader=test_loader,
                        save_txt=True,
                        save_txt_path=report_base_path + "_predictions.txt",
                    )                    
                    if y_true and y_pred:
                        print(f"   📊 Generating comprehensive evaluation report with confusion matrices...")
                        print(f"   📈 Evaluation will include:")
                        print(f"      • Classification report with precision, recall, F1-score")
                        print(f"      • Raw counts confusion matrix")
                        print(f"      • Normalized confusion matrix (by true labels - recall)")
                        print(f"      • Normalized confusion matrix (by predicted labels - precision)")
                        print(f"   💾 All visualizations will be saved to: {report_base_path}_*.png")
                        
                        # Display quick stats in console first
                        display_confusion_matrix_stats(y_true, y_pred, final_class_names)
                        
                        # Generate full report with visualizations
                        report_classification(y_true, y_pred, final_class_names, save_path_base=report_base_path)
                        
                        print(f"   ✅ Evaluation completed! Check the following files:")
                        print(f"      📄 Text report: {report_base_path}_report.txt")
                        print(f"      📊 Raw confusion matrix: {report_base_path}_cm_raw.png")
                        print(f"      📈 Recall confusion matrix: {report_base_path}_cm_norm_true.png")
                        print(f"      📉 Precision confusion matrix: {report_base_path}_cm_norm_pred.png")
                    else:
                        print("   ⚠️ Skipping evaluation report due to inference issues.")
                        
                except Exception as e:
                    print(f"⚠️ Error during evaluation: {e}")
                    traceback.print_exc()

            # --- Generate Grad-CAM Visualizations ---
            if test_loader and final_test_dataset:
                print(f"\n🔥 Generating Grad-CAM visualizations for fine-tuned classifier: {model_name}")
                gradcam_save_dir = os.path.join(model_results_dir, "gradcam_visualizations")
                
                # Prepare model for GradCAM
                model.train()  # Set to training mode for gradients
                for param in model.parameters():
                    param.requires_grad_(True)
                
                try:
                    generate_and_save_gradcam_per_class(
                        model=model,
                        dataset=final_test_dataset,
                        save_dir=gradcam_save_dir,
                        model_config=model_config,
                        device=device,
                        cam_algorithm='gradcam',
                        debug_layers=True
                    )
                except Exception as e:
                    print(f"⚠️ Could not generate Grad-CAM: {e}")
                    traceback.print_exc()

            print(f"\n✅ Classifier fine-tuning completed for model: {model_name}")
            print(f"   Results saved in: {model_results_dir}")

        else:
            print(f"\n❌ Model '{model_name}' classifier fine-tuning failed.")

        # Cleanup
        if 'model' in locals(): del model
        if 'train_datasets_list' in locals(): del train_datasets_list
        if 'val_datasets_list' in locals(): del val_datasets_list
        if 'test_datasets_list' in locals(): del test_datasets_list
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    print("\n🎉 All classifier fine-tuning completed.")


if __name__ == "__main__":
    main()
