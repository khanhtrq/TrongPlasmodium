# Plasmodium Classification Project

This repository contains scripts for training and evaluating deep learning models for Plasmodium (malaria parasite) classification in red blood cell images.

## Features

*   Supports loading models from both `torchvision` and `timm`.
*   Configurable training loop via YAML files (`config.yaml`, `config_local.yaml`).
*   Includes training components:
    *   Data loading from annotation files.
    *   Various optimizers (Adam, AdamW, SGD).
    *   Learning rate schedulers (StepLR, ReduceLROnPlateau, CosineAnnealingLR).
    *   Loss functions (CrossEntropyLoss with optional class weighting, FocalLoss, F1Loss).
    *   Automatic Mixed Precision (AMP) support.
    *   Gradient Clipping.
    *   Early Stopping.
*   Evaluation metrics: Classification report, confusion matrices (raw and normalized).
*   Visualization: Sample images, training curves, Grad-CAM heatmaps.
*   Multi-GPU training support using `DataParallel`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd PlasmodiumClassification-1
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # (Make sure requirements.txt includes torch, torchvision, timm, pyyaml, scikit-learn, matplotlib, numpy, tqdm, opencv-python)
    ```
3.  **Prepare your dataset:**
    *   Organize images under a root directory (e.g., `cropped_RBCs`).
    *   Create annotation files (`train_annotation.txt`, `val_annotation.txt`, `test_annotation.txt`) in the format: `relative/path/to/image.jpg label_index`.
4.  **Configure:**
    *   Edit `config.yaml` (for general/remote runs) or `config_local.yaml` (for local runs).
    *   Set `data_dir`, `root_dataset_dir`, `class_names`, `model_names`, and other training parameters.

## Usage

Run the main training script:

```bash
python main.py
# (Ensure the correct config file is selected within main.py if not using the default)
```

Results, logs, model weights, and visualizations will be saved in the directory specified by `results_dir` in the config file, organized by model name.
