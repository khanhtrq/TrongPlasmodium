import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import pprint  # Import pprint for pretty printing

from src.data_loader import AnnotationDataset, collate_fn_skip_error
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
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        model_names = config['model_names']

        # Access nested keys correctly
        training_params = config.get('training', {})
        optimizer_config = config.get('optimizer', {})
        scheduler_config = config.get('scheduler', {})
        device_config = config.get('device', {})

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
        class_names = config.get('class_names', None)

        # --- Validation ---
        if not model_names: raise ValueError("❌ 'model_names' cannot be empty in config.")
        if not isinstance(model_names, list): raise ValueError("❌ 'model_names' should be a list in config.")
        if not class_names: raise ValueError("❌ 'class_names' must be defined in config.")
        if not isinstance(class_names, list): raise ValueError("❌ 'class_names' should be a list in config.")
        if not results_dir: raise ValueError("❌ 'results_dir' must be defined in config.")

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
    num_classes = len(class_names)
    print(f"\n📋 Class Information:")
    print(f"   Number of classes: {num_classes}")
    print(f"   Class names: {class_names}")

    # --- Default Data Transformations ---
    default_input_size = 224
    default_transform = transforms.Compose([
        transforms.Resize((default_input_size, default_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"\n🔄 Default Transform Pipeline (used if model-specific is unavailable):")
    print(default_transform)

    # --- Dataset Paths ---
    train_annotation = os.path.join(data_dir, 'train_annotation.txt')
    val_annotation = os.path.join(data_dir, 'val_annotation.txt')
    test_annotation = os.path.join(data_dir, 'test_annotation.txt')
    print("\n💾 Dataset Annotation Files:")
    print(f"   Train: {train_annotation}")
    print(f"   Validation: {val_annotation}")
    print(f"   Test: {test_annotation}")
    print(f"   Image Root: {root_dataset_dir}")

    # --- Training and Evaluation Loop ---
    for model_name in model_names:
        print(f"\n{'='*25} Processing Model: {model_name} {'='*25}")

        # --- Create Results Directory for this Model ---
        model_results_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)
        print(f"   Results will be saved in: {model_results_dir}")

        # --- Initialize Model and Get Model-Specific Config ---
        try:
            model, input_size, model_specific_transform, model_config = initialize_model(
                model_name, num_classes=num_classes, use_pretrained=True, feature_extract=False
            )
        except Exception as e:
            print(f"❌ Critical Error initializing model '{model_name}': {e}")
            print("   Skipping this model.")
            continue

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

        # --- Load Datasets ---
        try:
            print("\n⏳ Loading datasets...")
            train_dataset = AnnotationDataset(train_annotation, root_dataset_dir, transform=transform_train, class_names=class_names)
            val_dataset = AnnotationDataset(val_annotation, root_dataset_dir, transform=transform_eval, class_names=class_names)
            test_dataset = AnnotationDataset(test_annotation, root_dataset_dir, transform=transform_eval, class_names=class_names)
            print("   Datasets loaded successfully.")

            plot_class_distribution_with_ratios(train_dataset, title="Training Set Class Distribution")
            analyze_class_distribution_across_splits({
                'Train': train_dataset,
                'Validation': val_dataset,
                'Test': test_dataset
            })

        except FileNotFoundError as e:
            print(f"❌ Error: Dataset file not found: {e}. Check 'data_dir' and annotation file names in config.")
            print("   Skipping this model.")
            continue
        except ValueError as e:
            print(f"❌ Error loading dataset: {e}. Check annotation file format or class name consistency.")
            print("   Skipping this model.")
            continue
        except Exception as e:
            print(f"❌ An unexpected error occurred while loading datasets: {e}")
            print("   Skipping this model.")
            continue

        # --- Create DataLoaders ---
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=num_workers > 0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=num_workers > 0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=num_workers > 0)
        print(f"\n📦 DataLoaders created (Batch size: {batch_size}, Workers: {num_workers})")

        dataloaders = {'train': train_loader, 'val': val_loader}

        # --- Visualize Sample Images ---
        plot_sample_images_per_class(train_dataset, num_samples=min(5, batch_size), model_config=model_config)

        # --- Model Setup ---
        model = model.to(device)
        if not hasattr(model, 'num_classes'):
            warnings.warn(f"Model {model_name} might not have a standard 'num_classes' attribute after initialization.")
            try:
                final_layer = model.get_classifier()
                if hasattr(final_layer, 'out_features'):
                    model.num_classes = final_layer.out_features
                    print(f"   Inferred num_classes from classifier: {model.num_classes}")
                else:
                    model.num_classes = num_classes
            except AttributeError:
                model.num_classes = num_classes

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
        elif criterion_name == 'crossentropy':
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
        print(f"\n🏋️ Starting training for model: {model_name}...")
        model_save_path = os.path.join(model_results_dir, f'{model_name}_best.pth')
        log_save_path = os.path.join(model_results_dir, f'{model_name}_training_log.csv')

        try:
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
        except Exception as train_err:
            print(f"❌❌❌ An error occurred during training for model {model_name}: {train_err}")
            import traceback
            traceback.print_exc()
            print("   Skipping evaluation and GradCAM for this model.")
            continue

        # --- Plot Training Curves ---
        plot_file_path = os.path.join(model_results_dir, f'{model_name}_training_curves.png')
        if history:
            try:
                plot_training_curves(history, title_suffix=f"({model_name})", save_path=plot_file_path)
            except Exception as e:
                print(f"⚠️ Could not plot/save training curves for {model_name}: {e}")
        else:
            print(f"⚠️ Skipping plotting for {model_name} due to empty or invalid history data.")

        # --- Evaluation on Test Set ---
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

        # --- Grad-CAM Generation ---
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
        del model, optimizer, criterion, scheduler, history
        if device.type == 'cuda': torch.cuda.empty_cache()

    print("\n🎉 All models processed.")

if __name__ == "__main__":
    main()
