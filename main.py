import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms, datasets  # Ensure datasets is imported
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import pprint  # Import pprint for pretty printing
import gc  # Import garbage collector

from src.data_loader import AnnotationDataset, ImageFolderWrapper, collate_fn_skip_error  # Import ImageFolderWrapper
from src.device_handler import get_device
from src.model_initializer import initialize_model
from src.training import train_model
from src.evaluation import infer_from_annotation, report_classification
from src.gradcam import generate_and_save_gradcam_per_class
from src.loss import FocalLoss, F1Loss, compute_class_weights
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
        data_dir = config['data_dir']
        root_dataset_dir = config['root_dataset_dir']
        initial_batch_size = config['batch_size']  # Store the initial batch size
        num_workers = config['num_workers']
        model_names = config['model_names']

        # Access nested keys correctly
        training_params = config.get('training', {})
        optimizer_config = config.get('optimizer', {})
        scheduler_config = config.get('scheduler', {})
        device_config = config.get('device', {})
        dataset_cfg = config.get('dataset_config', {})
        dataset_type = dataset_cfg.get('type', 'annotation').lower()  # Default to annotation

        # Annotation specific paths (relative to data_dir)
        annotation_train_file = dataset_cfg.get('annotation_train', 'train_annotation.txt')
        annotation_val_file = dataset_cfg.get('annotation_val', 'val_annotation.txt')
        annotation_test_file = dataset_cfg.get('annotation_test', 'test_annotation.txt')
        annotation_root = dataset_cfg.get('annotation_root', root_dataset_dir)  # Use specific root or fallback

        # ImageFolder specific paths
        imagefolder_root = dataset_cfg.get('imagefolder_root')
        imagefolder_train_subdir = dataset_cfg.get('imagefolder_train_subdir', 'train')
        imagefolder_val_subdir = dataset_cfg.get('imagefolder_val_subdir', 'val')
        imagefolder_test_subdir = dataset_cfg.get('imagefolder_test_subdir', 'test')

        num_epochs = training_params.get('num_epochs', 50)  # Provide default if missing
        patience = training_params.get('patience', 10)
        use_amp = training_params.get('use_amp', True)
        clip_grad_norm = float(training_params.get('clip_grad_norm', 1.0))

        optimizer_type = optimizer_config.get('type', 'Adam').lower()
        learning_rate = float(optimizer_config.get('lr', 1e-3))  # Add 'lr' under optimizer
        optimizer_params = optimizer_config.get('params', {})

        scheduler_type = scheduler_config.get('type', 'StepLR').lower()
        step_size = scheduler_config.get('step_size', 7)  # Default step_size
        gamma = scheduler_config.get('gamma', 0.1)  # Default gamma

        criterion_name = config.get('criterion', 'CrossEntropyLoss').lower()
        criterion_params = config.get('criterion_params', {})

        results_dir = config['results_dir']
        use_cuda = device_config.get('use_cuda', True)
        multi_gpu = device_config.get('multi_gpu', True)
        class_names = config.get('class_names', None)  # Keep class_names for annotation mapping

        # --- Validation ---
        if not model_names: raise ValueError("❌ 'model_names' cannot be empty in config.")
        if not isinstance(model_names, list): raise ValueError("❌ 'model_names' should be a list in config.")
        if dataset_type == 'imagefolder' and not imagefolder_root:
            raise ValueError("❌ 'imagefolder_root' must be defined in config when dataset_config.type is 'imagefolder'.")
        if dataset_type == 'annotation' and (not annotation_root):
            warnings.warn("⚠️ 'annotation_root' not explicitly defined for 'annotation' dataset type. Using 'root_dataset_dir' as fallback.")
            annotation_root = root_dataset_dir  # Fallback if needed
        if dataset_type == 'annotation' and not class_names:
            raise ValueError("❌ 'class_names' must be defined in config for 'annotation' dataset type.")
        if dataset_type == 'annotation' and not isinstance(class_names, list):
            raise ValueError("❌ 'class_names' should be a list in config for 'annotation' dataset type.")

    except KeyError as e:
        print(f"❌ Error: Missing key in configuration file: {e}")
        print("   Please ensure your config.yaml matches the expected structure.")
        exit()
    except (TypeError, ValueError) as e:
        print(f"❌ Error: Invalid value type or missing key in configuration file: {e}")
        print("   Please check the data types and structure in config.yaml.")
        exit()

    # --- Device Setup ---
    device, gpu_count = get_device(use_cuda, multi_gpu)
    print(f"\n🖥️ Device Selected: {device}")
    if device.type == 'cuda':
        print(f"   Number of GPUs available/requested: {gpu_count}")
        if gpu_count > 0:
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        if gpu_count == 0 and use_cuda:
            print("   ⚠️ CUDA requested but not available. Falling back to CPU.")
    elif device.type == 'cpu':
        print("   Running on CPU.")

    # --- Class Information ---
    if class_names:
        num_classes = len(class_names)
        print(f"\n📋 Class Information (from config):")
        print(f"   Number of classes: {num_classes}")
        print(f"   Class names: {class_names}")
    else:
        num_classes = None  # Will be determined after dataset loading if using ImageFolder
        print(f"\n📋 Class Information: Class names not provided in config, will be inferred from dataset if possible.")

    # --- Default Data Transformations ---
    default_input_size = 224
    default_transform = transforms.Compose([
        transforms.Resize((default_input_size, default_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"\n🔄 Default Transform Pipeline (used if model-specific is unavailable):")
    print(default_transform)

    # --- Dataset Paths (Informational Print) ---
    if dataset_type == 'annotation':
        train_annotation = os.path.join(data_dir, annotation_train_file)
        val_annotation = os.path.join(data_dir, annotation_val_file)
        test_annotation = os.path.join(data_dir, annotation_test_file)
        print("\n💾 Dataset Type: Annotation")
        print(f"   Train Annot: {train_annotation}")
        print(f"   Val Annot: {val_annotation}")
        print(f"   Test Annot: {test_annotation}")
        print(f"   Image Root: {annotation_root}")
    elif dataset_type == 'imagefolder':
        train_dir = os.path.join(imagefolder_root, imagefolder_train_subdir)
        val_dir = os.path.join(imagefolder_root, imagefolder_val_subdir)
        test_dir = os.path.join(imagefolder_root, imagefolder_test_subdir)
        print("\n💾 Dataset Type: ImageFolder")
        print(f"   Train Dir: {train_dir}")
        print(f"   Val Dir: {val_dir}")
        print(f"   Test Dir: {test_dir}")
    else:
        print(f"❌ Error: Unknown dataset type '{dataset_type}' in config.")
        exit()

    # --- Training and Evaluation Loop ---
    for model_name in model_names:
        print(f"\n{'='*25} Processing Model: {model_name} {'='*25}")

        # --- Create Results Directory for this Model ---
        model_results_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)
        print(f"   Results will be saved in: {model_results_dir}")

        # --- OOM Retry Loop ---
        current_batch_size = initial_batch_size  # Reset batch size for each new model
        model_trained_successfully = False
        history = None  # Initialize history outside the loop

        while current_batch_size >= 1:  # Retry until batch size is too small
            print(f"\n🔄 Attempting training for '{model_name}' with batch size: {current_batch_size}")
            try:
                # --- Initialize Model and Get Model-Specific Config ---
                temp_num_classes = num_classes if num_classes is not None else 1  # Placeholder if inferring
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

                # --- Load Datasets (Conditional Logic) ---
                print("\n⏳ Loading datasets...")
                try:
                    if dataset_type == 'annotation':
                        train_dataset = AnnotationDataset(train_annotation, annotation_root, transform=transform_train, class_names=class_names)
                        val_dataset = AnnotationDataset(val_annotation, annotation_root, transform=transform_eval, class_names=class_names)
                        test_dataset = AnnotationDataset(test_annotation, annotation_root, transform=transform_eval, class_names=class_names)
                        loaded_class_names = train_dataset.classes
                    elif dataset_type == 'imagefolder':
                        train_dataset = ImageFolderWrapper(root=train_dir, transform=transform_train)
                        val_dataset = ImageFolderWrapper(root=val_dir, transform=transform_eval)
                        test_dataset = ImageFolderWrapper(root=test_dir, transform=transform_eval)
                        loaded_class_names = train_dataset.classes
                        if class_names and loaded_class_names != class_names:
                            warnings.warn(f"⚠️ Class names inferred by ImageFolder ({loaded_class_names}) differ from config ({class_names}). Using inferred names.")
                    else:
                        raise ValueError(f"Invalid dataset_type '{dataset_type}' during dataset loading.")

                    if num_classes is None:
                        num_classes = len(loaded_class_names)
                        class_names = loaded_class_names
                        print(f"   Inferred {num_classes} classes: {class_names}")
                        if temp_num_classes != num_classes:
                            print(f"   Re-initializing model '{model_name}' with inferred {num_classes} classes...")
                            del model
                            if device.type == 'cuda': torch.cuda.empty_cache()
                            model, input_size, model_specific_transform, model_config = initialize_model(
                                model_name, num_classes=num_classes, use_pretrained=True, feature_extract=False
                            )

                    print("   Datasets loaded successfully.")

                    if current_batch_size == initial_batch_size:
                        plot_class_distribution_with_ratios(train_dataset, title="Training Set Class Distribution")
                        analyze_class_distribution_across_splits({
                            'Train': train_dataset,
                            'Validation': val_dataset,
                            'Test': test_dataset
                        })
                        plot_sample_images_per_class(train_dataset, num_samples=min(5, current_batch_size), model_config=model_config)

                except FileNotFoundError as e:
                    print(f"❌ Error: Dataset file/directory not found: {e}.")
                    print("   Check paths in config.yaml ('data_dir', 'annotation_*', 'imagefolder_*').")
                    print("   Skipping this model attempt.")
                    break
                except ValueError as e:
                    print(f"❌ Error loading dataset: {e}.")
                    print("   Check annotation file format, class name consistency, or image folder structure.")
                    print("   Skipping this model attempt.")
                    break
                except Exception as e:
                    print(f"❌ An unexpected error occurred while loading datasets: {e}")
                    print("   Skipping this model attempt.")
                    break

                # --- Create DataLoaders ---
                train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=num_workers > 0)
                val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=num_workers > 0)
                test_loader = DataLoader(test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=num_workers > 0)
                print(f"\n📦 DataLoaders created (Batch size: {current_batch_size}, Workers: {num_workers})")

                dataloaders = {'train': train_loader, 'val': val_loader}

                # --- Model Setup ---
                model = model.to(device)
                if device.type == 'cuda' and gpu_count > 1 and multi_gpu:
                    print(f"⚡ Enabling DataParallel across {gpu_count} GPUs for model '{model_name}'")
                    model = nn.DataParallel(model)

                # --- Define Optimizer ---
                params_to_update = model.parameters()
                if config.get('feature_extract', False):
                    print("   Optimizing only classifier parameters.")
                    params_to_update = [p for p in model.parameters() if p.requires_grad]
                else:
                    print("   Optimizing all model parameters.")

                print(f"\n🔧 Optimizer: {optimizer_type.capitalize()}")
                if optimizer_type == 'adam':
                    optimizer = optim.Adam(params_to_update, lr=learning_rate, **optimizer_params)
                elif optimizer_type == 'adamw':
                    optimizer = optim.AdamW(params_to_update, lr=learning_rate, **optimizer_params)
                elif optimizer_type == 'sgd':
                    optimizer = optim.SGD(params_to_update, lr=learning_rate, **optimizer_params)
                else:
                    print(f"⚠️ Unknown optimizer type '{optimizer_type}'. Defaulting to Adam.")
                    optimizer = optim.Adam(params_to_update, lr=learning_rate)
                print(f"   Initial Learning Rate: {learning_rate}")
                print(f"   Optimizer Params: {optimizer_params}")

                # --- Define Loss Function ---
                print(f"\n📉 Criterion: {criterion_name.capitalize()}")
                if criterion_name == 'focalloss':
                    alpha_weights = compute_class_weights(train_loader, num_classes, device)
                    print(f"   Computed FocalLoss alpha (class weights): {alpha_weights.cpu().numpy()}")
                    criterion = FocalLoss(
                        alpha=alpha_weights,
                        gamma=float(criterion_params.get('gamma', 2.0)),
                        reduction=criterion_params.get('reduction', 'mean')
                    ).to(device)
                elif criterion_name == 'f1loss':
                    criterion = F1Loss(
                        num_classes=num_classes,
                        beta=float(criterion_params.get('beta', 1.0)),
                        epsilon=float(criterion_params.get('epsilon', 1e-7)),
                        reduction=criterion_params.get('reduction', 'mean')
                    ).to(device)
                elif criterion_name == 'crossentropyloss':
                    if criterion_params.get('use_class_weights', False):
                        ce_weights = compute_class_weights(train_loader, num_classes, device)
                        print(f"   Computed CrossEntropy weights: {ce_weights.cpu().numpy()}")
                        criterion = nn.CrossEntropyLoss(weight=ce_weights).to(device)
                    else:
                        criterion = nn.CrossEntropyLoss().to(device)
                else:
                    print(f"⚠️ Unknown criterion '{criterion_name}'. Defaulting to CrossEntropyLoss.")
                    criterion = nn.CrossEntropyLoss().to(device)

                # --- Define LR Scheduler ---
                print(f"\n📅 LR Scheduler: {scheduler_type.capitalize()}")
                if scheduler_type == 'steplr':
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                elif scheduler_type == 'reducelronplateau':
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=gamma, patience=patience // 2, verbose=True)
                elif scheduler_type == 'cosineannealinglr':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01)
                else:
                    print(f"⚠️ Unknown scheduler type '{scheduler_type}'. Defaulting to StepLR.")
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

                # --- Start Training ---
                print(f"\n🏋️ Starting training for model: {model_name} (Batch Size: {current_batch_size})...")
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
                    clip_grad_norm=clip_grad_norm
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
            except FileNotFoundError as e:
                print(f"❌ Error: Dataset file not found during retry: {e}. Check 'data_dir' and annotation file names.")
                print("   Skipping this model.")
                break
            except ValueError as e:
                print(f"❌ Error loading dataset during retry: {e}. Check annotation file format or class name consistency.")
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
                class_names=class_names,
                device=device,
                dataloader=test_loader
            )

            if y_true and y_pred:
                report_base_path = os.path.join(model_results_dir, f'{model_name}_test_eval')
                report_classification(y_true, y_pred, class_names, save_path_base=report_base_path)
            else:
                print("   ⚠️ Skipping evaluation report generation due to inference issues or empty results.")

            print(f"\n🔥 Generating Grad-CAM visualizations for: {model_name}")
            gradcam_save_dir = os.path.join(model_results_dir, "gradcam_visualizations")

            try:
                generate_and_save_gradcam_per_class(
                    model=model,
                    dataset=test_dataset,
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

        del model, train_dataset, val_dataset, test_dataset
        if 'train_loader' in locals(): del train_loader
        if 'val_loader' in locals(): del val_loader
        if 'test_loader' in locals(): del test_loader
        if 'optimizer' in locals(): del optimizer
        if 'criterion' in locals(): del criterion
        if 'scheduler' in locals(): del scheduler
        if 'history' in locals(): del history
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    print("\n🎉 All models processed.")

if __name__ == "__main__":
    main()
