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

from src.data_loader import AnnotationDataset, ImageFolderWrapper, CombinedDataset, collate_fn_skip_error  # Adjusted import
from src.device_handler import get_device, setup_model_for_training  # MODIFIED: Import setup_model_for_training
from src.model_initializer import initialize_model
from src.training import train_model
from src.evaluation import infer_from_annotation, report_classification
from src.gradcam import generate_and_save_gradcam_per_class
from src.loss import FocalLoss, F1Loss, compute_class_weights, get_criterion
from src.visualization import (
    plot_sample_images_per_class,
    plot_training_curves,
    plot_class_distribution_with_ratios,
    analyze_class_distribution_across_splits
)

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

def main():
    # --- Configuration Loading ---
    config_file = 'config.yaml'
    config = load_config(config_file)

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
    final_class_names = class_names
    num_classes = len(class_names) if class_names else None
    if final_class_names:
        print(f"\n📋 Class Information (from config):")
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
                    model_name, num_classes=temp_num_classes, use_pretrained=True, feature_extract=False
                )

                # --- Determine Transform to Use ---
                if model_specific_transform:
                    transform_train = model_specific_transform
                    transform_eval = model_specific_transform
                    print(f"   Using TIMM's recommended transforms (input size: {input_size}x{input_size}).")
                else:
                    print(f"   Using DEFAULT transforms (input size: {input_size}x{input_size}).")
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
                                current_train_dataset = AnnotationDataset(resolve_path(data_dir, ann_train_path), train_root, transform=transform_train, class_names=final_class_names)
                            if ann_val_path:
                                current_val_dataset = AnnotationDataset(resolve_path(data_dir, ann_val_path), val_root, transform=transform_eval, class_names=final_class_names)
                            if ann_test_path:
                                current_test_dataset = AnnotationDataset(resolve_path(data_dir, ann_test_path), test_root, transform=transform_eval, class_names=final_class_names)

                        elif dataset_type == 'imagefolder':
                            imgf_root = d_cfg.get('imagefolder_root')
                            if not imgf_root: raise ValueError(f"Source {i+1}: 'imagefolder_root' is required for type 'imagefolder'.")
                            
                            imgf_train_subdir = d_cfg.get('imagefolder_train_subdir') # Get value, could be None
                            imgf_val_subdir = d_cfg.get('imagefolder_val_subdir')   # Get value, could be None
                            imgf_test_subdir = d_cfg.get('imagefolder_test_subdir')  # Get value, could be None

                            if imgf_train_subdir: # Only proceed if subdir is not None
                                train_dir = os.path.join(imgf_root, imgf_train_subdir)
                                if os.path.isdir(train_dir):
                                    current_train_dataset = ImageFolderWrapper(root=train_dir, transform=transform_train)
                                    print(f"     Loaded ImageFolder train from: {train_dir}")
                                else:
                                    warnings.warn(f"     ImageFolder train directory not found: {train_dir} (from source {i+1})")
                            else:
                                print(f"     Skipping ImageFolder train for source {i+1} as subdir is null.")

                            if imgf_val_subdir: # Only proceed if subdir is not None
                                val_dir = os.path.join(imgf_root, imgf_val_subdir)
                                if os.path.isdir(val_dir):
                                    current_val_dataset = ImageFolderWrapper(root=val_dir, transform=transform_eval)
                                    print(f"     Loaded ImageFolder val from: {val_dir}")
                                else:
                                    warnings.warn(f"     ImageFolder val directory not found: {val_dir} (from source {i+1})")
                            else:
                                print(f"     Skipping ImageFolder val for source {i+1} as subdir is null.")

                            if imgf_test_subdir: # Only proceed if subdir is not None
                                test_dir = os.path.join(imgf_root, imgf_test_subdir)
                                if os.path.isdir(test_dir):
                                    current_test_dataset = ImageFolderWrapper(root=test_dir, transform=transform_eval)
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
                                        model, input_size, model_specific_transform, model_config = initialize_model(model_name, num_classes=num_classes, use_pretrained=True, feature_extract=False)
                                        if model_specific_transform: transform_train = transform_eval = model_specific_transform
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
                         warnings.warn("⚠️ Cannot plot sample images: No training datasets loaded.")

                train_loader = DataLoader(final_train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=num_workers > 0)
                val_loader = DataLoader(final_val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=num_workers > 0) if final_val_dataset else None
                test_loader = DataLoader(final_test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=num_workers > 0) if final_test_dataset else None

                print(f"\n📦 DataLoaders created (Batch size: {current_batch_size}, Workers: {num_workers})")
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
                optimizer = optim.Adam(params_to_update, lr=learning_rate)

                print(f"\n📉 Criterion: {config.get('criterion', 'CrossEntropyLoss').capitalize()}")
                # Tính cls_num_list: số lượng phần tử của từng class trong tập train
                if hasattr(final_train_dataset_full, 'targets') and hasattr(final_train_dataset_full, 'classes'):
                    targets_np = np.array(final_train_dataset_full.targets)
                    cls_num_list = [int(np.sum(targets_np == i)) for i in range(len(final_train_dataset_full.classes))]
                else:
                    warnings.warn("Không thể tính cls_num_list: Dataset không có thuộc tính 'targets' hoặc 'classes'. Sử dụng mặc định [0].")
                    cls_num_list = [1] * num_classes

                criterion_params = config.get('criterion_params', {})
                criterion_params['cls_num_list'] = cls_num_list

                criterion = get_criterion(
                    config.get('criterion', 'CrossEntropyLoss'),
                    num_classes=num_classes,
                    device=device,
                    criterion_params=criterion_params
                )
                

                print(f"\n📅 LR Scheduler: {scheduler_config.get('type', 'StepLR').capitalize()}")
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config.get('step_size', 7), gamma=scheduler_config.get('gamma', 0.1))

                print(f"\n🏋️ Starting training for model: {model_name} (Batch Size: {current_batch_size}, Train Ratio: {train_ratio:.2f})...")
                model_save_path = os.path.join(model_results_dir, f'{model_name}_best.pth')
                log_save_path = os.path.join(model_results_dir, f'{model_name}_training_log.csv')

                model, history = train_model(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    num_epochs=num_epochs,
                    patience=patience,
                    use_amp=use_amp,
                    save_path=model_save_path,
                    log_path=log_save_path,
                    clip_grad_norm=clip_grad_norm,
                    train_ratio=train_ratio
                )

                model_trained_successfully = True
                print(f"✅ Training completed successfully for '{model_name}' with batch size {current_batch_size}.")
                break

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"\n🔥🔥🔥 CUDA Out of Memory detected for '{model_name}' with batch size {current_batch_size}! 🔥🔥🔥")
                    del model, train_loader, val_loader, test_loader, optimizer, criterion, scheduler
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

            print(f"\n🧪 Evaluating model: {model_name} on the test set using best weights...")
            model.eval()

            y_true, y_pred = infer_from_annotation(
                model=model,
                class_names=final_class_names,
                device=device,
                dataloader=test_loader
            )

            if y_true and y_pred:
                report_base_path = os.path.join(model_results_dir, f'{model_name}_test_eval')
                report_classification(y_true, y_pred, final_class_names, save_path_base=report_base_path)
            else:
                print("   ⚠️ Skipping evaluation report generation due to inference issues or empty results.")

            print(f"\n🔥 Generating Grad-CAM visualizations for: {model_name}")
            gradcam_save_dir = os.path.join(model_results_dir, "gradcam_visualizations")

            try:
                generate_and_save_gradcam_per_class(
                    model=model,
                    dataset=final_test_dataset,
                    save_dir=gradcam_save_dir,
                    model_config=model_config,
                    device=device
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


