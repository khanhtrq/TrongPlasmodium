import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler  # Adjusted imports
import torch.nn as nn
from torchvision import transforms, datasets  # Ensure datasets is imported
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import pprint  # Import pprint for pretty printing
import gc  # Import garbage collector
import math  # For ceiling function
import random  # For setting Python random seed

from src.data_loader import AnnotationDataset, ImageFolderWrapper, CombinedDataset, collate_fn_skip_error, create_weighted_random_sampler, get_effective_sampler_config  # Adjusted import
from src.device_handler import get_device, setup_model_for_training  # MODIFIED: Import setup_model_for_training
from src.model_initializer import initialize_model
from src.training import train_model, train_classifier_only  # MODIFIED: Import train_classifier_only
from src.evaluation import infer_from_annotation, report_classification
from src.regularizers import MaxNorm_via_PGD, Normalizer  # Import regularizers
from src.gradcam import generate_and_save_gradcam_per_class
from src.loss import FocalLoss, F1Loss, get_criterion  # MODIFIED: Removed unused compute_class_weights
# Advanced augmentation imports
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


def load_config(config_path='config.yaml'):
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

def main():
    # --- Configuration Loading ---
    config_file = 'config.yaml'
    config = load_config(config_file)

    # --- Set Seed for Reproducibility ---
    seed = config.get('seed', 42)  # Default to 42 if not specified
    set_seed(seed)

    # --- Print Loaded Configuration for Debugging ---
    print("\n" + "="*20 + " Loaded Configuration " + "="*20)
    pprint.pprint(config, indent=2)  # Use pprint for readable output
    print("="*60 + "\n")

    # --- Extract Configuration Parameters ---
    try:
        data_dir = config['data_dir']  # Base directory
        initial_batch_size = config['batch_size']
        num_workers = config['num_workers']
        model_names = config['model_names']

        datasets_config_list = config.get('datasets', [])  # Get the list of dataset configs
        if not datasets_config_list:
            raise ValueError("❌ 'datasets' list cannot be empty in config.")

        training_params = config.get('training', {})
        optimizer_config = config.get('optimizer', {})
        scheduler_config = config.get('scheduler', {})
        device_config = config.get('device', {})
        
        # --- Class Remapping Configuration (NEW) ---
        class_remapping_config = config.get('class_remapping', {})
        if class_remapping_config.get('enabled', False):
            print(f"\n🔄 Class remapping is enabled: {class_remapping_config.get('mapping', {})}")

        # --- Training Parameters ---
        dropout_rate = float(training_params.get('dropout_rate', 0.0))  # Ensure dropout_rate is a float
        num_epochs = training_params.get('num_epochs', 50)
        patience = training_params.get('patience', 10)
        use_amp = training_params.get('use_amp', True)
        clip_grad_norm = float(training_params.get('clip_grad_norm', 1.0))
        train_ratio = float(training_params.get('train_ratio', 1.0))  # Get train ratio
        if not (0.0 < train_ratio <= 1.0):
            warnings.warn(f"⚠️ Invalid train_ratio ({train_ratio}). Clamping to 1.0.")
            train_ratio = 1.0

        class_names = config.get('class_names', None)  # Still useful for initial mapping/consistency check
        learning_rate = float(optimizer_config.get('lr', 1e-4))  # Ensure lr is a float

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
        if gpu_count == 0 and device_config.get('use_cuda', True):
            print("   ⚠️ CUDA requested but not available. Falling back to CPU.")
    elif device.type == 'cpu':
        print("   Running on CPU.")

    # --- Class Information ---
    # Use remapped class names if remapping is enabled and final_class_names is provided
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

    # --- Default Data Transformations ---
    default_input_size = 224
    default_transform = transforms.Compose([
        transforms.Resize((default_input_size, default_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"\n🔄 Default Transform Pipeline (used if model-specific is unavailable):")
    print(default_transform)

    # --- Dataset Paths (Informational Print - Adjusted) ---
    print("\n💾 Defined Dataset Sources:")
    for i, d_cfg in enumerate(datasets_config_list):
        print(f"   Source {i+1}: Type = {d_cfg.get('type', 'N/A')}")

    # --- Training and Evaluation Loop ---
    for model_name in model_names:
        print(f"\n{'='*25} Processing Model: {model_name} {'='*25}")

        # --- Create Results Directory for this Model ---
        model_results_dir = os.path.join(config['results_dir'], model_name)
        os.makedirs(model_results_dir, exist_ok=True)
        print(f"   Results will be saved in: {model_results_dir}")

        # --- OOM Retry Loop ---
        current_batch_size = initial_batch_size
        model_trained_successfully = False
        history = None

        while current_batch_size >= 1:
            print(f"\n🔄 Attempting training for '{model_name}' with batch size: {current_batch_size}")
            try:
                # --- Initialize Model and Get Model-Specific Config ---
                temp_num_classes = num_classes if num_classes is not None else 1
                model, input_size, model_specific_transform, model_config = initialize_model(
                    model_name, num_classes=temp_num_classes, use_pretrained=True, feature_extract=False, drop_rate=dropout_rate
                )                # --- Determine Transform to Use ---
                # Check if augmentation is enabled in config
                aug_config = config.get('augmentation', {})
                if aug_config.get('enabled', False):
                    # Use advanced augmentation strategy
                    print(f"   🎨 Using ADVANCED augmentation strategy: {aug_config.get('strategy', 'light')}")
                    aug_strategy = create_augmentation_strategy(config, model_config)
                    transform_train = aug_strategy.get_train_transform()
                    transform_eval = aug_strategy.get_eval_transform()
                    print(f"   📈 Train augmentation: {len(transform_train.transforms)} steps")
                    print(f"   📊 Eval transforms: {len(transform_eval.transforms)} steps")
                    
                    # Print augmentation details
                    print(f"   🔧 Augmentation details:")
                    print(f"      Strategy: {aug_strategy.strategy}")
                    print(f"      Input size: {aug_strategy.input_size}")
                    print(f"      Interpolation: {aug_strategy.interpolation}")
                elif model_specific_transform:
                    # Use timm's recommended transforms if augmentation is disabled
                    transform_train = model_specific_transform
                    transform_eval = model_specific_transform

                else:
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

                # Print transforms being used
                print("\n" + "="*60)
                print("TRANSFORMS CONFIGURATION")
                print("="*60)
                print(f"📊 Model: {model_name}")
                print(f"🖼️  Input size: {input_size}x{input_size}")
                print(f"🔧 Augmentation enabled: {aug_config.get('enabled', False)}")
                print(f"📦 Using model-specific transform: {model_specific_transform is not None}")
                
                print("\n🏋️  TRAINING TRANSFORMS:")
                if hasattr(transform_train, 'transforms'):
                    for i, t in enumerate(transform_train.transforms, 1):
                        print(f"   {i}. {t}")
                else:
                    print(f"   Single transform: {transform_train}")
                
                print("\n🔍 EVALUATION TRANSFORMS:")
                if hasattr(transform_eval, 'transforms'):
                    for i, t in enumerate(transform_eval.transforms, 1):
                        print(f"   {i}. {t}")
                else:
                    print(f"   Single transform: {transform_eval}")
                
                if aug_config.get('enabled', False):
                    print(f"\n⚙️  AUGMENTATION DETAILS:")
                    print(f"   Strategy: {aug_config.get('strategy', 'basic')}")
                    print(f"   Config: {aug_config}")
                
                print("="*60 + "\n")
                
                # --- Load Multiple Datasets ---
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
                                return os.path.join(base, p) if p and not os.path.isabs(p) else p

                            if ann_train_path:
                                current_train_dataset = AnnotationDataset(resolve_path(data_dir, ann_train_path), train_root, transform=transform_train, class_names=final_class_names, class_remapping=class_remapping_config)
                            if ann_val_path:
                                current_val_dataset = AnnotationDataset(resolve_path(data_dir, ann_val_path), val_root, transform=transform_eval, class_names=final_class_names, class_remapping=class_remapping_config)
                            if ann_test_path:
                                current_test_dataset = AnnotationDataset(resolve_path(data_dir, ann_test_path), test_root, transform=transform_eval, class_names=final_class_names, class_remapping=class_remapping_config)

                        elif dataset_type == 'imagefolder':
                            imgf_root = d_cfg.get('imagefolder_root')
                            if not imgf_root: raise ValueError(f"Source {i+1}: 'imagefolder_root' is required for type 'imagefolder'.")
                            
                            imgf_train_subdir = d_cfg.get('imagefolder_train_subdir') # Get value, could be None
                            imgf_val_subdir = d_cfg.get('imagefolder_val_subdir')   # Get value, could be None
                            imgf_test_subdir = d_cfg.get('imagefolder_test_subdir')  # Get value, could be None

                            if imgf_train_subdir: # Only proceed if subdir is not None
                                train_dir = os.path.join(imgf_root, imgf_train_subdir)
                                if os.path.isdir(train_dir):
                                    current_train_dataset = ImageFolderWrapper(root=train_dir, transform=transform_train, class_remapping=class_remapping_config)
                                    print(f"     Loaded ImageFolder train from: {train_dir}")
                                else:
                                    warnings.warn(f"     ImageFolder train directory not found: {train_dir} (from source {i+1})")
                            else:
                                print(f"     Skipping ImageFolder train for source {i+1} as subdir is null.")

                            if imgf_val_subdir: # Only proceed if subdir is not None
                                val_dir = os.path.join(imgf_root, imgf_val_subdir)
                                if os.path.isdir(val_dir):
                                    current_val_dataset = ImageFolderWrapper(root=val_dir, transform=transform_eval, class_remapping=class_remapping_config)
                                    print(f"     Loaded ImageFolder val from: {val_dir}")
                                else:
                                    warnings.warn(f"     ImageFolder val directory not found: {val_dir} (from source {i+1})")
                            else:
                                print(f"     Skipping ImageFolder val for source {i+1} as subdir is null.")

                            if imgf_test_subdir: # Only proceed if subdir is not None
                                test_dir = os.path.join(imgf_root, imgf_test_subdir)
                                if os.path.isdir(test_dir):
                                    current_test_dataset = ImageFolderWrapper(root=test_dir, transform=transform_eval, class_remapping=class_remapping_config)
                                    print(f"     Loaded ImageFolder test from: {test_dir}")
                                else:
                                    warnings.warn(f"     ImageFolder test directory not found: {test_dir} (from source {i+1})")
                            else:
                                print(f"     Skipping ImageFolder test for source {i+1} as subdir is null.")
                        else:
                            raise ValueError(f"Source {i+1}: Invalid dataset_type '{dataset_type}'.")

                        dataset_to_check = current_train_dataset or current_val_dataset or current_test_dataset
                        if dataset_to_check:
                            current_classes = dataset_to_check.classes
                            if not first_dataset_loaded:
                                if final_class_names is None:
                                    final_class_names = current_classes
                                    num_classes = len(final_class_names)
                                    print(f"   Inferred {num_classes} classes from first dataset: {final_class_names}")
                                    if temp_num_classes != num_classes:
                                        print(f"   Re-initializing model '{model_name}' with inferred {num_classes} classes...")
                                        del model; gc.collect(); torch.cuda.empty_cache()
                                        model, input_size, model_specific_transform, model_config = initialize_model(model_name, num_classes=num_classes, use_pretrained=True, feature_extract=False, drop_rate=dropout_rate)
                                        if model_specific_transform: 
                                            transform_train = transform_eval = model_specific_transform
                                        else:
                                            transform_train = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize(mean=model_config['mean'], std=model_config['std'])])
                                            transform_eval = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=model_config['mean'], std=model_config['std'])])
                                elif final_class_names != current_classes:
                                    raise ValueError(f"Source {i+1}: Class name mismatch! Expected {final_class_names} but found {current_classes}.")
                                first_dataset_loaded = True
                            elif final_class_names != current_classes:
                                raise ValueError(f"Source {i+1}: Class name mismatch! Expected {final_class_names} but found {current_classes}.")

                            if current_train_dataset: train_datasets_list.append(current_train_dataset)
                            if current_val_dataset: val_datasets_list.append(current_val_dataset)
                            if current_test_dataset: test_datasets_list.append(current_test_dataset)
                        else:
                            print(f"   Source {i+1}: No valid dataset splits found.")

                    except Exception as e:
                        print(f"❌ Error loading dataset source {i+1}: {e}")
                        raise e

                if not train_datasets_list:
                    raise ValueError("❌ No training datasets were loaded successfully.")
                if not val_datasets_list:
                    warnings.warn("⚠️ No validation datasets were loaded. Validation phase will be skipped.")
                if not test_datasets_list:
                    warnings.warn("⚠️ No test datasets were loaded. Final evaluation will be skipped.")

                if len(train_datasets_list) > 1:
                    print(f"   Combining {len(train_datasets_list)} training datasets using CombinedDataset wrapper...")
                    try:
                        final_train_dataset_full = CombinedDataset(train_datasets_list)  # Use CombinedDataset here
                    except (AttributeError, ValueError) as e:
                        print(f"❌ Error creating CombinedDataset: {e}. Check dataset attributes ('classes', 'targets').")
                        raise e  # Halt if combination fails
                    print(f"   Combined training set size: {len(final_train_dataset_full)}")
                else:
                    final_train_dataset_full = train_datasets_list[0]
                    print(f"   Using single training dataset (size: {len(final_train_dataset_full)})")

                if train_ratio < 1.0:
                    subset_size = math.ceil(len(final_train_dataset_full) * train_ratio)
                    indices = list(range(subset_size))
                    final_train_dataset = Subset(final_train_dataset_full, indices)
                    if not hasattr(final_train_dataset, 'classes'): final_train_dataset.classes = final_class_names
                    if not hasattr(final_train_dataset, 'targets'):
                        warnings.warn("Cannot easily get 'targets' from Subset for analysis when train_ratio < 1.0.")
                        final_train_dataset.targets = []
                else:
                    final_train_dataset = final_train_dataset_full
                    print(f"   Using full training set (ratio: {train_ratio:.2f}).")

                final_val_dataset = val_datasets_list[0] if val_datasets_list else None
                final_test_dataset = test_datasets_list[0] if test_datasets_list else None

                print("   Datasets finalized.")

                if current_batch_size == initial_batch_size:
                    print("\n📊 Analyzing Full Training Set Distribution:")
                    if hasattr(final_train_dataset_full, 'targets') and hasattr(final_train_dataset_full, 'classes') and final_train_dataset_full.targets:
                        plot_class_distribution_with_ratios(final_train_dataset_full, title="Full Training Set Class Distribution")
                    else:
                        warnings.warn("⚠️ Cannot plot full training set distribution: Underlying dataset missing 'targets' or 'classes' attribute, or targets list is empty.")

                    if final_val_dataset and final_test_dataset:
                        train_set_for_analysis = final_train_dataset_full
                        if hasattr(train_set_for_analysis, 'targets') and hasattr(train_set_for_analysis, 'classes'):
                            analyze_class_distribution_across_splits({
                                'Train': train_set_for_analysis,
                                'Validation': final_val_dataset,
                                'Test': final_test_dataset
                            })
                        else:
                            warnings.warn("⚠️ Cannot analyze splits: Training dataset object missing 'targets' or 'classes'.")

                    if train_datasets_list:
                        first_train_ds = train_datasets_list[0]
                        if hasattr(first_train_ds, 'classes') and first_train_ds.classes and \
                           hasattr(first_train_ds, 'imgs') and hasattr(first_train_ds, 'loader') and hasattr(first_train_ds, 'transform'):
                            print("   Plotting sample images from the *first* loaded training dataset.")
                            plot_sample_images_per_class(first_train_ds, num_samples=min(5, current_batch_size), model_config=model_config)
                        else:
                            warnings.warn("⚠️ Cannot plot sample images: First training dataset object missing required attributes ('classes', 'imgs', 'loader', 'transform').")
                    else:
                         warnings.warn("⚠️ Cannot plot sample images: No training datasets loaded.")                # --- Create WeightedRandomSampler if enabled ---
                sampler_config = config.get('weighted_random_sampler', {})
                train_sampler = None
                if sampler_config.get('enabled', False):
                    try:
                        train_sampler = create_weighted_random_sampler(
                            final_train_dataset, 
                            num_classes, 
                            sampler_config
                        )
                        print(f"   ✅ WeightedRandomSampler created for training data")
                    except Exception as e:
                        print(f"   ⚠️ Failed to create WeightedRandomSampler: {e}")
                        print(f"   📝 Falling back to standard random sampling")
                        train_sampler = None

                # --- Create DataLoaders ---
                train_loader = DataLoader(
                    final_train_dataset, 
                    batch_size=current_batch_size, 
                    sampler=train_sampler,  # Use WeightedRandomSampler if available
                    shuffle=(train_sampler is None),  # Only shuffle if no sampler is used
                    num_workers=num_workers, 
                    pin_memory=True, 
                    collate_fn=collate_fn_skip_error, 
                    persistent_workers=num_workers > 0, 
                    drop_last=True
                )
                val_loader = DataLoader(final_val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=num_workers > 0, drop_last=True) if final_val_dataset else None
                test_loader = DataLoader(final_test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=num_workers > 0, drop_last=True) if final_test_dataset else None                
                
                sampler_info = "WeightedRandomSampler" if train_sampler else "Random Shuffle"
                print(f"\n📦 DataLoaders created (Batch size: {current_batch_size}, Workers: {num_workers}, Sampling: {sampler_info})")
                dataloaders = {'train': train_loader}
                if val_loader: dataloaders['val'] = val_loader

                # --- Setup Model for Training Device(s) ---
                model, device_actual = setup_model_for_training(  # MODIFIED: Use setup_model_for_training
                    model,
                    use_cuda=device_config.get('use_cuda', True),
                    multi_gpu=device_config.get('multi_gpu', True)
                )
                print(f"   Model is now on device: {device_actual}")

                params_to_update = model.parameters()
                if config.get('feature_extract', False):
                    print("   Optimizing only classifier parameters.")
                    params_to_update = [p for p in model.parameters() if p.requires_grad]
                else:
                    print("   Optimizing all model parameters.")

                print(f"\n🔧 Optimizer: {optimizer_config.get('type', 'Adam').capitalize()}")
                
                optimizer_type = optimizer_config.get('type', 'Adam').lower()
                optimizer_params_config = optimizer_config.get('params') # Get value, could be None
                if optimizer_params_config is None: # Ensure it's a dict
                    optimizer_params_config = {}
                
                # Common parameter: learning rate
                optimizer_params_config['lr'] = learning_rate

                if optimizer_type == 'adam':
                    optimizer = optim.Adam(params_to_update, **optimizer_params_config)
                elif optimizer_type == 'adamw':
                    optimizer = optim.AdamW(params_to_update, **optimizer_params_config)
                elif optimizer_type == 'sgd':
                    optimizer = optim.SGD(params_to_update, **optimizer_params_config)
                # Add more optimizers as needed
                # elif optimizer_type == 'rmsprop':
                #     optimizer = optim.RMSprop(params_to_update, **optimizer_params_config)
                else:
                    warnings.warn(f"Unsupported optimizer type: {optimizer_type}. Defaulting to Adam.")
                    optimizer = optim.Adam(params_to_update, lr=learning_rate)
                
                # Extract first_stage_epochs from training params
                first_stage_epochs = training_params.get('first_stage_epochs', 0)
                
                # Check for dual criterion configuration
                criterion_a_name = config.get('criterion_a', config.get('criterion', 'CrossEntropyLoss')).lower()
                criterion_a_params = config.get('criterion_a_params')
                if criterion_a_params is None:
                    criterion_a_params = config.get('criterion_params') # Try fallback
                    if criterion_a_params is None: # Ensure it's a dict
                        criterion_a_params = {}
                
                criterion_b_name = config.get('criterion_b', '').lower()
                criterion_b_params = config.get('criterion_b_params') # Get value, could be None
                if criterion_b_params is None: # Ensure it's a dict
                    criterion_b_params = {}
                
                # Determine if we're using dual criterions
                using_dual_criterions = first_stage_epochs > 0 and criterion_b_name
                
                # Calculate cls_num_list for both criterions if needed
                if hasattr(final_train_dataset_full, 'targets') and hasattr(final_train_dataset_full, 'classes'):
                    targets_np = np.array(final_train_dataset_full.targets)
                    cls_num_list = [int(np.sum(targets_np == i)) for i in range(len(final_train_dataset_full.classes))]
                else:
                    warnings.warn("Cannot calculate cls_num_list: Dataset missing 'targets' or 'classes' attributes. Using default [1] * num_classes.")
                    cls_num_list = [1] * num_classes
                
                # Add class list to both criterion params if criterion is ldamloss
                if criterion_a_name == 'ldamloss':
                    criterion_a_params['cls_num_list'] = cls_num_list
                if criterion_b_name == 'ldamloss':
                    criterion_b_params['cls_num_list'] = cls_num_list
                
                # Add samples_per_cls to both criterion params if criterion is cbloss
                if criterion_a_name in ['cbloss', 'classbalancedloss']:
                    criterion_a_params['samples_per_cls'] = cls_num_list
                    criterion_a_params['num_classes'] = num_classes
                if criterion_b_name in ['cbloss', 'classbalancedloss']:
                    criterion_b_params['samples_per_cls'] = cls_num_list
                    criterion_b_params['num_classes'] = num_classes

                # Print criterion setup information
                if using_dual_criterions:
                    print(f"\n📉 Dual Criterion Training:")
                    print(f"   First {first_stage_epochs} epochs: {criterion_a_name}")
                    print(f"   Remaining epochs: {criterion_b_name}")
                else:
                    print(f"\n📉 Single Criterion: {criterion_a_name}")
                
                # Initialize criterion_a (always needed)
                criterion_a = get_criterion(
                    criterion_a_name,
                    num_classes=num_classes,
                    device=device,
                    criterion_params=criterion_a_params
                )
                
                # Initialize criterion_b (only if using dual criterions)
                criterion_b = None
                if using_dual_criterions:
                    criterion_b = get_criterion(
                        criterion_b_name,
                        num_classes=num_classes, 
                        device=device,
                        criterion_params=criterion_b_params
                    )
                
                # Add noise_sigma parameters to optimizer if needed
                if criterion_a_name in ['bmcloss', 'gailoss']:
                    optimizer.add_param_group({'params': criterion_a.noise_sigma, 'lr': 1e-2, 'name': 'noise_sigma_a'})
                if criterion_b and criterion_b_name in ['bmcloss', 'gailoss']:
                    optimizer.add_param_group({'params': criterion_b.noise_sigma, 'lr': 1e-2, 'name': 'noise_sigma_b'})
                
                print(f"\n📅 LR Scheduler: {scheduler_config.get('type', 'StepLR').capitalize()}")
                scheduler_type = scheduler_config.get('type', 'StepLR').lower()
                scheduler = None
                if scheduler_type == 'cosineannealinglr':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=num_epochs,
                        eta_min=scheduler_config.get('min_lr', 0)
                    )
                elif scheduler_type == 'reducelronplateau':
                    # Remove keys not accepted by ReduceLROnPlateau
                    reduce_params = dict(scheduler_config)
                    reduce_params.pop('type', None)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode=reduce_params.get('mode', 'max'),
                        factor=reduce_params.get('factor', 0.1),
                        patience=reduce_params.get('patience', 10),
                        threshold=reduce_params.get('threshold', 1e-4),
                        min_lr=reduce_params.get('min_lr', 0),
                        verbose=True
                    )
                elif scheduler_type == 'steplr':
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=scheduler_config.get('step_size', 7),
                        gamma=scheduler_config.get('gamma', 0.1)
                    )                
                else:
                    warnings.warn(f"Unsupported scheduler type: {scheduler_type}. Defaulting to StepLR.")
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=7,
                        gamma=0.1
                    )
                print(f"\n🏋️ Starting training for model: {model_name} (Batch Size: {current_batch_size}, Train Ratio: {train_ratio:.2f})...")
                model_save_path = os.path.join(model_results_dir, f'{model_name}_best.pth')
                log_save_path = os.path.join(model_results_dir, f'{model_name}_training_log.csv')

                # --- Setup MixUp/CutMix if enabled ---
                mixup_fn = None
                aug_config = config.get('augmentation', {})
                if aug_config.get('enabled', False):
                    mixup_alpha = aug_config.get('mixup_alpha', 0)
                    cutmix_alpha = aug_config.get('cutmix_alpha', 0)
                    
                    if mixup_alpha > 0 or cutmix_alpha > 0:
                        print(f"   🎭 Initializing MixUp/CutMix (mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha})")
                        mixup_fn = MixupCutmixWrapper(
                            mixup_alpha=mixup_alpha,
                            cutmix_alpha=cutmix_alpha,
                            cutmix_minmax=None,
                            prob=aug_config.get('mixup_cutmix_prob', 1.0),
                            switch_prob=aug_config.get('switch_prob', 0.5),
                            mode='batch',
                            label_smoothing=aug_config.get('label_smoothing', 0.1),
                            num_classes=num_classes
                        )
                        if mixup_fn.is_enabled():
                            print(f"      ✅ MixUp/CutMix enabled successfully")
                            print(f"      📊 Label smoothing: {aug_config.get('label_smoothing', 0.1)}")
                            print(f"      🎲 Switch probability: {aug_config.get('switch_prob', 0.5)}")
                        else:
                            print(f"      ⚠️ MixUp/CutMix could not be enabled (timm not available)")
                            mixup_fn = None
                    else:
                        print(f"   📋 MixUp/CutMix disabled (both alpha values are 0)")                # --- Initialize Regularizers for Main Training ---
                print(f"\n🔧 Initializing regularizers for main training...")
                main_regularizer_config = config.get('regularization', {})
                max_norm_reg, tau_norm_reg, tau_freq = init_regularizers(main_regularizer_config)

                model, history, train_best_val_metric = train_model(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion_a,  # Pass criterion_a as primary criterion
                    criterion_b=criterion_b,  # Pass criterion_b as secondary criterion
                    first_stage_epochs=first_stage_epochs,  # Pass first_stage_epochs parameter
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    num_epochs=num_epochs,
                    patience=patience,
                    use_amp=use_amp,
                    save_path=model_save_path,
                    log_path=log_save_path,
                    clip_grad_norm=clip_grad_norm,
                    train_ratio=train_ratio,
                    mixup_fn=mixup_fn,  # Pass the MixUp/CutMix function
                    max_norm_regularizer=max_norm_reg,  # Pass max-norm regularizer
                    tau_normalizer=tau_norm_reg,  # Pass tau-normalization regularizer
                    tau_norm_frequency=tau_freq  # Pass tau-normalization frequency
                )

                model_trained_successfully = True
                print(f"✅ Training completed successfully for '{model_name}' with batch size {current_batch_size}.")
                break

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"\n🔥🔥🔥 CUDA Out of Memory detected for '{model_name}' with batch size {current_batch_size}! 🔥🔥🔥")
                    del model, train_loader, val_loader, test_loader, optimizer, criterion_a, criterion_b, scheduler
                    if 'history' in locals(): del history
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        print("   CUDA cache cleared.")

                    new_batch_size = current_batch_size // 2
                    print(f"   Reducing batch size from {current_batch_size} to {new_batch_size}.")
                    current_batch_size = new_batch_size

                    if current_batch_size < 1:
                        print(f"❌ Batch size reduced to {current_batch_size}. Cannot train. Skipping model '{model_name}'.")
                        break
                else:
                    print(f"❌❌❌ An unexpected RuntimeError occurred during training setup or execution for model {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    print("   Skipping this model.")
                    break
            except Exception as e:
                print(f"❌❌❌ An unexpected error occurred during training setup or execution for model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                print("   Skipping this model.")
                break

        if model_trained_successfully:
            plot_file_path = os.path.join(model_results_dir, f'{model_name}_training_curves.png')
            if history:
                try:
                    plot_training_curves(history, title_suffix=f"({model_name}, BS={current_batch_size})", save_path=plot_file_path)
                except Exception as e:
                    print(f"⚠️ Could not plot/save training curves for {model_name}: {e}")
            else:
                print(f"⚠️ Skipping plotting for {model_name} due to empty or invalid history data.")

            # --- Classifier-Only Training Stage ---
            classifier_train_config = config.get('classifier_only_training', {})
            print(f"\n🔍 Classifier-Only Fine-tuning Config: {classifier_train_config}")
            if classifier_train_config.get('enabled', False):
                print(f"\n🚀 Starting Classifier-Only Fine-tuning for '{model_name}'...")

                # 1. Prepare model: Freeze ALL parameters first, then unfreeze only the classifier.
                print("   Freezing all model parameters initially.")
                for param in model.parameters():
                    param.requires_grad = False

                # Get the classifier module using the model's method
                classifier_module = None
                if hasattr(model, 'get_classifier') and callable(model.get_classifier):
                    try:
                        classifier_module = model.get_classifier()
                        if isinstance(classifier_module, nn.Module):
                            print(f"   Unfreezing classifier parameters obtained from model.get_classifier()")
                            # print(f"   Re-initializing classifier weights with Xavier uniform initialization.")
                            # torch.nn.init.xavier_uniform_(classifier_module.weight.data)  # Initialize weights
                            for param in classifier_module.parameters():
                                param.requires_grad = True
                        else:
                            warnings.warn(f"   ⚠️ model.get_classifier() did not return an nn.Module. NO layers were un-frozen. Ensure get_classifier() is implemented correctly.")
                            classifier_module = None # Reset if not a module
                    except Exception as e_gc:
                        warnings.warn(f"   ⚠️ Error calling model.get_classifier(): {e_gc}. NO layers were un-frozen.")
                        classifier_module = None
                else:
                    warnings.warn("   ⚠️ model.get_classifier() method not found. NO layers were un-frozen for classifier training. Implement this method in your model class.")

                # Collect parameters that are now trainable (should only be classifier ones)
                classifier_params_to_train = [p for p in model.parameters() if p.requires_grad]
                
                if not classifier_params_to_train:
                    warnings.warn(f"⚠️ CRITICAL: No trainable parameters found for classifier-only training of '{model_name}'. Skipping this phase. Check 'get_classifier()' implementation in your model class.")
                else:
                    num_trainable = sum(p.numel() for p in classifier_params_to_train)
                    print(f"   Number of parameters for classifier fine-tuning: {num_trainable}")

                    # 2. Optimizer for classifier training (ensure it uses only classifier_params_to_train)
                    cls_optimizer_config = classifier_train_config.get('optimizer', {'type': 'Adam', 'params': {'lr': 1e-4}})
                    cls_optimizer_type = cls_optimizer_config.get('type', 'Adam').lower()
                    cls_optimizer_params_config = cls_optimizer_config.get('params', {})
                    if not isinstance(cls_optimizer_params_config, dict):
                        warnings.warn(f"Classifier optimizer 'params' is not a dictionary. Using default {{'lr': 1e-4}}. Check config.")
                        cls_optimizer_params_config = {'lr': 1e-4}

                    if cls_optimizer_type == 'adam':
                        optimizer_cls = optim.Adam(classifier_params_to_train, **cls_optimizer_params_config)
                    elif cls_optimizer_type == 'adamw':
                        optimizer_cls = optim.AdamW(classifier_params_to_train, **cls_optimizer_params_config)
                    else:
                        warnings.warn(f"Unsupported classifier optimizer type: {cls_optimizer_type}. Defaulting to Adam.")
                        optimizer_cls = optim.Adam(classifier_params_to_train, lr=1e-4)
                    
                    print(f"   Classifier Optimizer: {type(optimizer_cls).__name__} with params: {optimizer_cls.defaults}")

                    # 3. Scheduler for classifier training
                    cls_scheduler_config = classifier_train_config.get('scheduler', scheduler_config)  # Use main scheduler_config as fallback
                    cls_scheduler_type = cls_scheduler_config.get('type', 'ReduceLROnPlateau').lower()
                    scheduler_cls = None
                    if cls_scheduler_type == 'cosineannealinglr':
                        scheduler_cls = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer_cls,
                            T_max=classifier_train_config.get('num_epochs', 10),
                            eta_min=cls_scheduler_config.get('min_lr', 0)
                        )
                    elif cls_scheduler_type == 'reducelronplateau':
                        # Remove keys not accepted by ReduceLROnPlateau - same as main training
                        reduce_params = dict(cls_scheduler_config)
                        reduce_params.pop('type', None)
                        scheduler_cls = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer_cls,
                            mode=reduce_params.get('mode', 'max'),
                            factor=reduce_params.get('factor', 0.1),
                            patience=reduce_params.get('patience', 10),
                            threshold=reduce_params.get('threshold', 1e-4),
                            min_lr=reduce_params.get('min_lr', 0),
                            verbose=True
                        )
                    elif cls_scheduler_type == 'steplr':
                        scheduler_cls = optim.lr_scheduler.StepLR(
                            optimizer_cls,
                            step_size=cls_scheduler_config.get('step_size', 7),
                            gamma=cls_scheduler_config.get('gamma', 0.1)
                        )
                    else:
                        warnings.warn(f"Unsupported classifier scheduler type: {cls_scheduler_type}. Defaulting to ReduceLROnPlateau.")
                        scheduler_cls = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer_cls,
                            mode='max',
                            factor=0.1,
                            patience=10,
                            verbose=True
                        )

                    if scheduler_cls:
                        print(f"   Classifier Scheduler: {type(scheduler_cls).__name__} with config: {cls_scheduler_config}")
                    else:
                        print("   Classifier Scheduler: None")
                    
                    # 4. Criterion for classifier training
                    # --- BEGIN: Load classifier-only criterion config properly ---
                    # Use classifier_only_training's criterion config if present, else fallback to main
                    cls_criterion_a_name = classifier_train_config.get('criterion_a')
                    if cls_criterion_a_name is None: # Try legacy 'criterion' within classifier_config
                        cls_criterion_a_name = classifier_train_config.get('criterion')
                    if cls_criterion_a_name is None: # Fallback to main config's 'criterion_a'
                        cls_criterion_a_name = config.get('criterion_a')
                    if cls_criterion_a_name is None: # Fallback to main config's legacy 'criterion'
                        cls_criterion_a_name = config.get('criterion', 'CrossEntropyLoss') # Default if nothing found
                    cls_criterion_a_name = str(cls_criterion_a_name).lower()

                    cls_criterion_a_params = classifier_train_config.get('criterion_a_params')
                    if cls_criterion_a_params is None: # Try legacy 'criterion_params' within classifier_config
                        cls_criterion_a_params = classifier_train_config.get('criterion_params')
                    if cls_criterion_a_params is None: # Fallback to main config's 'criterion_a_params'
                        cls_criterion_a_params = config.get('criterion_a_params')
                    if cls_criterion_a_params is None: # Fallback to main config's legacy 'criterion_params'
                        cls_criterion_a_params = config.get('criterion_params', {}) # Default to empty dict
                    if cls_criterion_a_params is None: # Ensure it's a dict if all fallbacks result in None
                        cls_criterion_a_params = {}
                    
                    print(f"   Classifier Criterion A: {cls_criterion_a_name} with params: {cls_criterion_a_params}")

                    cls_criterion_b_name = classifier_train_config.get('criterion_b')
                    if cls_criterion_b_name is None: # Fallback to main config's 'criterion_b'
                        cls_criterion_b_name = config.get('criterion_b', '') # Default to empty string
                    cls_criterion_b_name = str(cls_criterion_b_name).lower() if cls_criterion_b_name else ''

                    cls_criterion_b_params = classifier_train_config.get('criterion_b_params')
                    if cls_criterion_b_params is None: # Fallback to main config's 'criterion_b_params'
                        cls_criterion_b_params = config.get('criterion_b_params', {}) # Default to empty dict
                    if cls_criterion_b_params is None: # Ensure it's a dict
                        cls_criterion_b_params = {}
                    
                    # first_stage_epochs for classifier-only phase
                    cls_first_stage_epochs = classifier_train_config.get('first_stage_epochs', 0)
                    # --- END: Load classifier-only criterion config properly ---

                    # Calculate cls_num_list for both criterions if needed
                    if hasattr(final_train_dataset_full, 'targets') and hasattr(final_train_dataset_full, 'classes'):
                        targets_np = np.array(final_train_dataset_full.targets)
                        cls_num_list = [int(np.sum(targets_np == i)) for i in range(len(final_train_dataset_full.classes))]
                    else:
                        warnings.warn("Cannot calculate cls_num_list: Dataset missing 'targets' or 'classes' attributes. Using default [1] * num_classes.")
                        cls_num_list = [1] * num_classes

                    # Add class list to both criterion params if criterion is ldamloss
                    if cls_criterion_a_name == 'ldamloss':
                        cls_criterion_a_params['cls_num_list'] = cls_num_list
                    if cls_criterion_b_name == 'ldamloss':
                        cls_criterion_b_params['cls_num_list'] = cls_num_list

                    # Add samples_per_cls to both criterion params if criterion is cbloss
                    if cls_criterion_a_name in ['cbloss', 'classbalancedloss']:
                        cls_criterion_a_params['samples_per_cls'] = cls_num_list
                        cls_criterion_a_params['num_classes'] = num_classes
                    if cls_criterion_b_name in ['cbloss', 'classbalancedloss']:
                        cls_criterion_b_params['samples_per_cls'] = cls_num_list
                        cls_criterion_b_params['num_classes'] = num_classes

                    # Print criterion setup information
                    using_cls_dual_criterions = cls_first_stage_epochs > 0 and cls_criterion_b_name
                    if using_cls_dual_criterions:
                        print(f"\n📉 Classifier-Only Dual Criterion Training:")
                        print(f"   First {cls_first_stage_epochs} epochs: {cls_criterion_a_name}")
                        print(f"   Remaining epochs: {cls_criterion_b_name}")
                    else:
                        print(f"\n📉 Classifier-Only Single Criterion: {cls_criterion_a_name}")

                    # Initialize classifier-only criterions
                    criterion_cls_a = get_criterion(
                        cls_criterion_a_name,
                        num_classes=num_classes,
                        device=device,
                        criterion_params=cls_criterion_a_params
                    )
                    criterion_cls_b = None
                    if using_cls_dual_criterions:
                        criterion_cls_b = get_criterion(
                            cls_criterion_b_name,
                            num_classes=num_classes,
                            device=device,
                            criterion_params=cls_criterion_b_params
                        )                    # Add noise_sigma parameters to optimizer if needed
                    if cls_criterion_a_name in ['bmcloss', 'gailoss']:
                        optimizer_cls.add_param_group({'params': criterion_cls_a.noise_sigma, 'lr': 1e-2, 'name': 'noise_sigma_a'})
                    if criterion_cls_b and cls_criterion_b_name in ['bmcloss', 'gailoss']:
                        optimizer_cls.add_param_group({'params': criterion_cls_b.noise_sigma, 'lr': 1e-2, 'name': 'noise_sigma_b'})

                    cls_model_save_path = os.path.join(model_results_dir, f'{model_name}_classifier_best.pth')
                    cls_log_save_path = os.path.join(model_results_dir, f'{model_name}_classifier_training_log.csv')

                    # --- Create WeightedRandomSampler for classifier training if enabled ---
                    main_sampler_config = config.get('weighted_random_sampler', {})
                    cls_sampler_config_specific = classifier_train_config.get('weighted_random_sampler', {})
                    
                    # Get effective sampler config (classifier-specific overrides main config)
                    cls_effective_sampler_config = get_effective_sampler_config(main_sampler_config, cls_sampler_config_specific)
                    
                    cls_train_sampler = None
                    cls_dataloaders = dataloaders.copy()  # Start with existing dataloaders
                    
                    if cls_effective_sampler_config.get('enabled', False):
                        try:
                            cls_train_sampler = create_weighted_random_sampler(
                                final_train_dataset, 
                                num_classes, 
                                cls_effective_sampler_config
                            )
                            print(f"   ✅ WeightedRandomSampler created for classifier training")
                            
                            # Create new training DataLoader with classifier-specific sampler
                            cls_train_loader = DataLoader(
                                final_train_dataset, 
                                batch_size=current_batch_size, 
                                sampler=cls_train_sampler,
                                shuffle=False,  # Cannot use shuffle with sampler
                                num_workers=num_workers, 
                                pin_memory=True, 
                                collate_fn=collate_fn_skip_error, 
                                persistent_workers=num_workers > 0, 
                                drop_last=True
                            )
                            cls_dataloaders['train'] = cls_train_loader
                            print(f"   📦 Created new training DataLoader for classifier training with WeightedRandomSampler")
                            
                        except Exception as e:
                            print(f"   ⚠️ Failed to create WeightedRandomSampler for classifier training: {e}")
                            print(f"   📝 Using existing training DataLoader")
                    else:
                        if cls_sampler_config_specific.get('enabled') is not None:
                            # Explicitly disabled for classifier training
                            print(f"   📝 WeightedRandomSampler explicitly disabled for classifier training")
                        else:
                            # Using main config (which might be enabled or disabled)
                            sampler_status = "enabled" if main_sampler_config.get('enabled', False) else "disabled"
                            print(f"   📝 Using main WeightedRandomSampler config for classifier training ({sampler_status})")

                    # --- Initialize Regularizers for Classifier Training ---
                    print(f"\n🔧 Initializing regularizers for classifier training...")
                    cls_regularizer_config = classifier_train_config.get('regularization', main_regularizer_config)
                    cls_max_norm_reg, cls_tau_norm_reg, cls_tau_freq = init_regularizers(cls_regularizer_config)
                    if mixup_alpha > 0 or cutmix_alpha > 0:
                        print(f"   🎭 MixUp/CutMix is enabled for classifier training with alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha}")
                        mixup_fn.enable = True
                    try:
                        model, history_cls = train_classifier_only(
                            model=model,
                            dataloaders=cls_dataloaders,  # Use potentially modified dataloaders
                            criterion=criterion_cls_a,
                            criterion_b=criterion_cls_b,
                            first_stage_epochs=cls_first_stage_epochs,
                            optimizer=optimizer_cls,
                            scheduler=scheduler_cls,
                            device=device,
                            num_epochs=classifier_train_config.get('num_epochs', 10),
                            patience=classifier_train_config.get('patience', 3),
                            use_amp=use_amp,
                            save_path=cls_model_save_path,
                            log_path=cls_log_save_path,
                            clip_grad_norm=classifier_train_config.get('clip_grad_norm', clip_grad_norm),
                            train_ratio=classifier_train_config.get('train_ratio', train_ratio),
                            init_best_val_metric=0.0,  # Pass initial best metric from main training
                            max_norm_regularizer=cls_max_norm_reg,  # Pass max-norm regularizer
                            tau_normalizer=cls_tau_norm_reg,  # Pass tau-normalization regularizer
                            tau_norm_frequency=cls_tau_freq,  # Pass tau-normalization frequency
                            mixup_fn=mixup_fn  # Pass the MixUp/CutMix function
                        )
                        print(f"✅ Classifier-Only Fine-tuning completed for '{model_name}'.")

                        cls_plot_file_path = os.path.join(model_results_dir, f'{model_name}_classifier_training_curves.png')
                        if history_cls:
                            plot_training_curves(history_cls, title_suffix=f" ({model_name} - Classifier, BS={current_batch_size})", save_path=cls_plot_file_path)
                        else:
                            print(f"⚠️ No history returned from classifier training for {model_name}, cannot plot curves.")
                    
                    except Exception as e_cls:
                        print(f"❌❌❌ An error occurred during classifier-only training for model {model_name}: {e_cls}")
                        import traceback
                        traceback.print_exc()
                        print("   Skipping further processing for this model after classifier training error.")
                    
                    del optimizer_cls, criterion_cls_a
                    if scheduler_cls: del scheduler_cls
                    if criterion_cls_b: del criterion_cls_b
                    if 'history_cls' in locals(): del history_cls
                    gc.collect()
                    if device.type == 'cuda': torch.cuda.empty_cache()

            print(f"\n🧪 Evaluating model: {model_name} on the test set using best weights...")
            model.eval()
            report_base_path = os.path.join(model_results_dir, f'{model_name}_test_eval')
            y_true, y_pred = infer_from_annotation(
                model=model,
                class_names=final_class_names,
                device=device,
                dataloader=test_loader,
                save_txt = True,
                save_txt_path=report_base_path + "_predictions.txt",
            )

            if y_true and y_pred:
                report_classification(y_true, y_pred, final_class_names, save_path_base=report_base_path)
            else:
                print("   ⚠️ Skipping evaluation report generation due to inference issues or empty results.")

            print(f"\n🔥 Generating Grad-CAM visualizations for: {model_name}")
            gradcam_save_dir = os.path.join(model_results_dir, "gradcam_visualizations")
            
            # CRITICAL: Properly prepare model for GradCAM
            print(f"🔧 Preparing model for GradCAM visualization...")
            
            # Load the best model weights
            best_model_path = os.path.join(model_results_dir, f'{model_name}_best.pth')
            if os.path.exists(best_model_path):
                print(f"   📂 Loading best model weights from: {best_model_path}")
                try:
                    # Load state dict properly
                    checkpoint = torch.load(best_model_path, map_location=device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    print(f"   ✅ Model weights loaded successfully")
                except Exception as e:
                    print(f"   ⚠️ Could not load model weights: {e}. Using current model state.")
            else:
                print(f"   ⚠️ Best model file not found: {best_model_path}. Using current model state.")
            
            # CRITICAL: Set model to training mode and enable gradients for GradCAM
            model.train()
            for param in model.parameters():
                param.requires_grad_(True)
            
            print(f"   ✅ Model set to training mode with gradients enabled for GradCAM")
            
            try:
                generate_and_save_gradcam_per_class(
                    model=model,
                    dataset=final_test_dataset,
                    save_dir=gradcam_save_dir,
                    model_config=model_config,
                    device=device,
                    cam_algorithm='gradcam',  # You can make this configurable
                    debug_layers=True  # Enable debugging for better layer selection
                )
            except Exception as e:
                print(f"⚠️ Could not generate Grad-CAM for {model_name}: {e}")
                import traceback
                traceback.print_exc()

            print(f"\n✅ Processing finished for model: {model_name}")
            print(f"   Results saved in: {model_results_dir}")

        else:
            print(f"\n❌ Model '{model_name}' could not be trained successfully (due to OOM or other errors).")

        del model, train_datasets_list, val_datasets_list, test_datasets_list
        if 'final_train_dataset' in locals(): del final_train_dataset
        if 'final_val_dataset' in locals(): del final_val_dataset
        if 'final_test_dataset' in locals(): del final_test_dataset
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    print("\n🎉 All models processed.")

if __name__ == "__main__":
    main()


