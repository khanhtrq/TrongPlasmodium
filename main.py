import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

from src.data_loader import AnnotationDataset
from src.device_handler import get_device
from src.model_initializer import initialize_model
from src.training import train_model
from src.evaluation import infer_from_annotation, report_classification
from src.gradcam import generate_and_save_gradcam_per_class
from src.loss import FocalLoss, compute_alpha_from_dataloader
import matplotlib.pyplot as plt
import matplotlib

from src.visualization import plot_sample_images_per_class, plot_training_curves

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        # Update optimizer learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

if __name__ == "__main__":
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data_dir = config['data_dir']
    root_dataset_dir = config['root_dataset_dir']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    model_names = config['model_names']
    num_epochs = config['num_epochs']
    patience = config['patience']
    learning_rate = float(config['learning_rate'])
    step_size = config['step_size']
    gamma = config['gamma']
    use_amp = config['use_amp']
    clip_grad_norm = float(config.get('clip_grad_norm', 0.5))
    results_dir = config['results_dir']
    tpu_available = config['tpu_available']
    
    # Get new configuration options with defaults
    use_focal_loss = config.get('use_focal_loss', False)
    focal_alpha = float(config.get('focal_alpha', 0.25))
    focal_gamma = float(config.get('focal_gamma', 2.0))
    use_warmup = config.get('use_warmup', False)
    warmup_epochs = int(config.get('warmup_epochs', 5))
    
    # GPU configuration
    use_cuda = config.get('use_cuda', True)
    multi_gpu = config.get('multi_gpu', True)
    
    # Device configuration with multi-GPU support
    device, gpu_count = get_device(use_cuda, multi_gpu)
    print(f"🖥️ Device: {device}")
    if torch.cuda.is_available() and use_cuda:
        print(f"🚀 Using {gpu_count} GPU{'s' if gpu_count > 1 else ''}")
        if gpu_count > 0:
            for i in range(gpu_count):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Get class names from config
    class_names = config.get('class_names', None)
    if class_names:
        print(f"📋 Using class names from config: {class_names}")
    else:
        print("⚠️ No class names found in config. Will use automatically generated class names.")

    # Data transformations
    default_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset paths
    train_annotation = os.path.join(data_dir, 'train_annotation.txt')
    val_annotation = os.path.join(data_dir, 'val_annotation.txt')
    test_annotation = os.path.join(data_dir, 'test_annotation.txt')

    # Training and evaluation
    for model_name in model_names:
        print(f"\n🚀 Training model: {model_name}")
        
        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize model and get transform from timm if available
        model, input_size, model_transform, model_config = initialize_model(model_name, len(class_names) if class_names else None)
        
        # Apply proper weight initialization for stability
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        # Only apply custom initialization to the classifier head to preserve pretrained weights
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            model.classifier.apply(init_weights)
        
        # Use transform from timm if available, default if not
        transform = model_transform if model_transform is not None else default_transform
        print(f"Using input size: {input_size}x{input_size}")
        print(f"Using transform pipeline from: {'timm' if model_transform else 'default config'}")
        
        # Dataset with model's transform
        train_dataset = AnnotationDataset(train_annotation, root_dataset_dir, transform, class_names)
        val_dataset = AnnotationDataset(val_annotation, root_dataset_dir, transform, class_names)
        test_dataset = AnnotationDataset(test_annotation, root_dataset_dir, transform, class_names)
        
        # Verify class names
        print(f"🏷️ Class names: {train_dataset.classes}")
        print(f"🔢 Number of classes: {len(train_dataset.classes)}")
        
        # Wrap into DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # Move model to device
        model = model.to(device)
        
        # Visualize samples with model-specific transforms
        plot_sample_images_per_class(train_dataset, num_samples=3)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Choose loss function based on config
        if use_focal_loss:
            print(f"📊 Using FocalLoss with alpha={focal_alpha}, gamma={focal_gamma}")
            alpha = compute_alpha_from_dataloader(train_loader, len(train_dataset.classes), device)
            criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)
        else:
            print("📊 Using standard CrossEntropyLoss")
            criterion = nn.CrossEntropyLoss()
        
        # Choose scheduler based on config
        if use_warmup:
            print(f"📈 Using warmup scheduler for {warmup_epochs} epochs")
            scheduler = WarmupCosineScheduler(
                optimizer, 
                warmup_epochs=warmup_epochs,
                total_epochs=num_epochs,
                base_lr=learning_rate
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        model_save_path = os.path.join(model_dir, 'best_model.pth')
        log_save_path = os.path.join(model_dir, 'training_log.csv')
        model, history = train_model(
            model, dataloaders, criterion, optimizer, scheduler, device,
            num_epochs=num_epochs, patience=patience, use_amp=use_amp, save_path=model_save_path,
            log_path=log_save_path, clip_grad_norm=clip_grad_norm
        )
        
        # Plotting
        plot_file_path = os.path.join(model_dir, 'training_curves.png')
        if history and all(k in history for k in ['train_loss', 'val_loss', 'train_acc_macro', 'val_acc_macro']):
            try:
                plot_training_curves(history, title_suffix=f"({model_name})", save_path=plot_file_path)
            except Exception as e:
                print(f"⚠️ Could not plot/save training curves for {model_name}: {e}")
        else:
            print(f"⚠️ Skipping plotting for {model_name} due to missing history data.")
        
        print(f"\n🔍 Inferencing with model: {model_name}")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            model.eval()
            
            # Use test_loader for efficient batch inference
            y_true, y_pred = infer_from_annotation(
                model=model, 
                annotation_file=test_annotation,
                class_names=train_dataset.classes, 
                root_dir=root_dataset_dir,
                device=device,
                transform=transform,
                input_size=(input_size, input_size),
                dataloader=test_loader
            )
        except Exception as e:
            print(f"❌ Error during inference for {model_name}: {e}")
            continue

        report_file_path = os.path.join(model_dir, 'classification_report.txt')
        with open(report_file_path, 'w') as f:
            f.write(classification_report(y_true, y_pred, target_names=train_dataset.classes, digits=4))
        
        cm_plot_base_path = os.path.join(model_dir, 'confusion_matrix')
        report_classification(y_true, y_pred, train_dataset.classes, save_path_base=cm_plot_base_path)
        
        print(f"\n🎯 Generating Grad-CAM for model: {model_name}")
        
        gradcam_save_dir = os.path.join(model_dir, "gradcam")
        os.makedirs(gradcam_save_dir, exist_ok=True)
        
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            model.eval()
            generate_and_save_gradcam_per_class(
                model_name=model_name,
                model=model,
                dataset=test_dataset,
                transform=transform,
                save_dir=gradcam_save_dir
            )
        except Exception as e:
            print(f"⚠️ Could not generate Grad-CAM for {model_name}: {e}")

        print(f"✅ Results for {model_name} saved to {model_dir}")
