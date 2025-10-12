# Plasmodium Classification Project

This repository contains scripts for training and evaluating deep learning models for Plasmodium (malaria parasite) classification in red blood cell images.

## Features

*   **Flexible Model Selection:** Supports both `torchvision` and `timm` models. Choose from EfficientNet, ResNet, GhostNet, ConvNeXt, and more by editing `model_names` in your config.
*   **Config-Driven Everything:** Training, augmentation, regularization, optimizer, scheduler, loss functions, and more—all controlled via YAML files (`config.yaml`, `tuning_classifier_config.yaml`).
*   **Class Remapping:** Merge, drop, or rename classes easily with the `class_remapping` section in your config. No more hardcoded class hacks.
*   **Data Augmentation:** From minimal to extreme. Use AutoAugment, RandAugment, MixUp, CutMix, color jitter, random erasing, and more. Tweak every parameter in the `augmentation` section.
*   **Regularization:** Max-norm, tau-normalization, label smoothing, and more. Because overfitting is for amateurs.
*   **Classifier-Only Training:** Fine-tune just the classifier head with separate settings, optimizer, scheduler, and loss. See `classifier_only_training` in your config.
*   **Weighted Sampling:** Handle class imbalance with `WeightedRandomSampler`—fully configurable.
*   **Training & Fine-Tuning:** Train from scratch, fine-tune, or load a pretrained checkpoint. Early stopping, AMP, gradient clipping, and more.
*   **Evaluation:** Confusion matrices, classification reports, Grad-CAM visualizations (manual hooks for full control), and error analysis.

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
    *   Organize images under a root directory (e.g., `cropped_RBCs` or as specified in `data_dir`).
    *   Create annotation files (`train_annotation.txt`, `val_annotation.txt`, `test_annotation.txt`) in the format: `relative/path/to/image.jpg label_index`.
    *   Or use an ImageFolder structure as shown in the config examples.
4.  **Configure:**
    *   Edit `config.yaml`, `config_local.yaml`, or `tuning_classifier_config.yaml`.
    *   Set `data_dir`, `datasets`, `class_names`, `class_remapping`, `model_names`, augmentation, optimizer, scheduler, and other training parameters.
    *   For classifier-only fine-tuning, adjust the `classifier_only_training` section.
    *   For device, set `device.use_cuda`.

## Usage

Run the main training script:

```bash
python main.py
# (Use the config file you want by editing main.py or passing as an argument if supported)
```

Or, if you just want to fine-tune the classifier head (and let the backbone chill), run:

```bash
python finetune_classifier.py
# (Make sure your config enables classifier_only_training)
```

Results, logs, model weights, and visualizations will be saved in the directory specified by `results_dir` in your config, organized by model name.

## Pro Tips

- **Class Remapping:** Want to merge or drop classes? Edit the `class_remapping` section. Your model, your rules.
- **Augmentation:** Tweak `augmentation.strategy` for more or less data madness. Try `extreme` if you’re feeling lucky.
- **Regularization:** Overfitting? Crank up `max_norm` or try tau-normalization.
- **Fine-Tuning:** Use `classifier_only_training` to squeeze extra juice from your backbone.
- **Pretrained Models:** Load checkpoints with the `pretrained_model` section for transfer learning or resuming training.

---

If you get stuck, blame YAML. Or just read the config comments—they’re more helpful than most StackOverflow answers.
