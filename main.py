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
from src.loss import FocalLoss
import matplotlib.pyplot as plt
import matplotlib

from src.visualization import plot_sample_images_per_class, plot_training_curves

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
learning_rate = config['learning_rate']
step_size = config['step_size']
gamma = config['gamma']
use_amp = config['use_amp']
results_dir = config['results_dir']
tpu_available = config['tpu_available']

# Device configuration
device = get_device()
print(f"Using device: {device}")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset paths
train_annotation = os.path.join(data_dir, 'train_annotation.txt')
val_annotation = os.path.join(data_dir, 'val_annotation.txt')
test_annotation = os.path.join(data_dir, 'test_annotation.txt')

# Load datasets
train_dataset = AnnotationDataset(train_annotation, root_dataset_dir, transform)
val_dataset = AnnotationDataset(val_annotation, root_dataset_dir, transform)
test_dataset = AnnotationDataset(test_annotation, root_dataset_dir, transform)

# --- Weighted Sampling Setup ---
sampler = None
print("⚖️ Calculating sample weights for balanced training...")
train_labels = np.array(train_dataset.targets)
sample_weights = compute_sample_weight(class_weight='balanced', y=train_labels)
sampler_weights = torch.DoubleTensor(sample_weights)
# sampler = WeightedRandomSampler(weights=sampler_weights, num_samples=len(sampler_weights), replacement=True)
print("✅ Sampler created.")
# --- End Weighted Sampling Setup ---

# Wrap into DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

plot_sample_images_per_class(train_dataset, num_samples=5)

dataloaders = {
    'train': train_loader,
    'val': val_loader
}

# Training and evaluation
for model_name in model_names:
    print(f"\n🚀 Training model: {model_name}")
    
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    model, input_size = initialize_model(model_name, len(train_dataset.classes))
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = FocalLoss(gamma=5.0, alpha=1).to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    model_save_path = os.path.join(model_dir, 'best_model.pth')
    log_save_path = os.path.join(model_dir, 'training_log.csv')
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, device,
        num_epochs=num_epochs, patience=patience, use_amp=use_amp, save_path=model_save_path,
        log_path=log_save_path
    )
    
    # --- Plotting ---
    plot_file_path = os.path.join(model_dir, 'training_curves.png')
    if history and all(k in history for k in ['train_loss', 'val_loss', 'train_acc_macro', 'val_acc_macro']):
        try:
            # Use the save_path parameter directly in the function call
            plot_training_curves(history, title_suffix=f"({model_name})", save_path=plot_file_path)
        except Exception as e:
            print(f"⚠️ Could not plot/save training curves for {model_name}: {e}")
    else:
        print(f"⚠️ Skipping plotting for {model_name} due to missing history data.")
    # --- End Plotting ---
    
    print(f"\n🔍 Inferencing with model: {model_name}")
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()
        y_true, y_pred = infer_from_annotation(model, test_annotation, train_dataset.classes, root_dataset_dir, device)
    except Exception as e:
        print(f"Error during inference setup or execution for {model_name}: {e}")
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
