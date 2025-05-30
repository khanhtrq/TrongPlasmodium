import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef
)
from pathlib import Path

from src.data_loader import AnnotationDataset, ImageFolderWrapper, CombinedDataset, collate_fn_skip_error
from src.model_initializer import initialize_model
from src.device_handler import get_device
from src.evaluation import report_classification

def score_diff_fn(max1, max2, threshold= 0.6353 -  0.3297):
    return max1 - max2 < threshold
    


class SimpleInference:
    """
    🎯 Simple and robust inference script for PlasmodiumClassification.
    
    Features:
    - Automatic config loading with fallbacks
    - Model discovery and selection
    - Dataset loading with remapping support
    - Comprehensive evaluation metrics
    - Clean result reporting
    """
    
    def __init__(self, config_path='config_local.yaml', verbose=True):
        self.config_path = config_path
        self.verbose = verbose
        self.config = None
        self.device = None
        self.model = None
        self.class_names = None
        self.num_classes = None
        self.model_config = None
        self.transform = None
        
        # Load configuration
        self._load_configuration()
        self._setup_device()
        
    def _load_configuration(self):
        """Load and validate configuration with fallbacks."""
        # Try multiple config file locations
        config_candidates = [
            self.config_path,
            'config_local.yaml',
            'config.yaml'
        ]
        
        config_loaded = False
        for config_file in config_candidates:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as file:
                        self.config = yaml.safe_load(file)
                    
                    if self.verbose:
                        print(f"📖 Configuration loaded from: {config_file}")
                    
                    self.config_path = config_file
                    config_loaded = True
                    break
                    
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Failed to load {config_file}: {e}")
                    continue
        
        if not config_loaded:
            raise FileNotFoundError(f"❌ No valid configuration file found. Tried: {config_candidates}")
        
        # Validate required configuration fields
        required_fields = ['data_dir', 'class_names']
        missing_fields = [field for field in required_fields if field not in self.config]
        
        if missing_fields:
            raise ValueError(f"❌ Missing required config fields: {missing_fields}")
        
        # Get class information with remapping support
        class_remapping_config = self.config.get('class_remapping', {})
        if class_remapping_config.get('enabled', False) and class_remapping_config.get('final_class_names'):
            self.class_names = class_remapping_config['final_class_names']
            if self.verbose:
                print(f"🔄 Using remapped class names: {self.class_names}")
        else:
            self.class_names = self.config.get('class_names', [])
            if self.verbose:
                print(f"📋 Using original class names: {self.class_names}")
        
        self.num_classes = len(self.class_names)
        
        if self.verbose:
            print(f"📊 Number of classes: {self.num_classes}")
            print(f"📂 Data directory: {self.config['data_dir']}")
    
    def _setup_device(self):
        """Setup computation device."""
        device_config = self.config.get('device', {})
        self.device, _ = get_device(
            device_config.get('use_cuda', True), 
            device_config.get('multi_gpu', False)  # Single GPU for inference
        )
        
        if self.verbose:
            print(f"🖥️ Using device: {self.device}")
    
    def discover_available_models(self, results_dir=None):
        """Discover available trained models."""
        if results_dir is None:
            results_dir = self.config.get('results_dir', 'results')
        
        available_models = {}
        
        if not os.path.exists(results_dir):
            if self.verbose:
                print(f"⚠️ Results directory not found: {results_dir}")
            return available_models
        
        for item in os.listdir(results_dir):
            model_dir = os.path.join(results_dir, item)
            if os.path.isdir(model_dir):
                # Look for .pth files
                checkpoints = []
                for file in os.listdir(model_dir):
                    if file.endswith('.pth'):
                        checkpoint_path = os.path.join(model_dir, file)
                        checkpoints.append(checkpoint_path)
                
                if checkpoints:
                    available_models[item] = checkpoints
        
        return available_models
    
    def select_model_checkpoint(self, model_checkpoint=None):
        """Select model checkpoint - now simplified to use direct path."""
        if model_checkpoint and os.path.exists(model_checkpoint):
            if self.verbose:
                print(f"✅ Using specified model: {model_checkpoint}")
            return model_checkpoint
        
        # Auto-discover models as fallback
        available_models = self.discover_available_models()
        
        if not available_models:
            raise FileNotFoundError("❌ No trained models found in results directory")
        
        if self.verbose:
            print(f"\n🔍 Available models:")
            for model_name, checkpoints in available_models.items():
                print(f"   📁 {model_name}:")
                for checkpoint in checkpoints:
                    print(f"      - {os.path.basename(checkpoint)}")
        
        # Auto-select best available model
        for model_name, checkpoints in available_models.items():
            for checkpoint in checkpoints:
                if '_best.pth' in checkpoint:
                    if self.verbose:
                        print(f"🚀 Auto-selected best model: {checkpoint}")
                    return checkpoint
        
        # Fallback to first available
        first_model = list(available_models.values())[0]
        selected = first_model[0]
        if self.verbose:
            print(f"🔄 Using first available model: {selected}")
        return selected
    
    def load_model(self, model_checkpoint, model_name=None):
        """Load trained model from checkpoint."""
        if self.verbose:
            print(f"\n🔄 Loading model from: {model_checkpoint}")
        
        # Extract model name if not provided
        if model_name is None:
            checkpoint_path = Path(model_checkpoint)
            if checkpoint_path.parent.name not in ['results', 'results_kaggle']:
                model_name = checkpoint_path.parent.name
            else:
                model_name = checkpoint_path.stem.replace('_best', '').replace('_classifier', '')
        
        if self.verbose:
            print(f"   📝 Model name: {model_name}")
        
        # Check if model_num_classes is specified in config
        config_model_classes = self.config.get('model_num_classes', None)
        if config_model_classes is not None:
            model_num_classes = config_model_classes
            if self.verbose:
                print(f"   🔧 Using configured model classes: {model_num_classes}")
                print(f"   📊 Dataset classes: {self.num_classes}")
        else:
            # Auto-detect from checkpoint
            try:
                checkpoint = torch.load(model_checkpoint, map_location='cpu')
                
                # Detect model's number of classes from checkpoint
                model_num_classes = None
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Look for classifier layer to determine number of classes
                for key, tensor in state_dict.items():
                    if 'classifier' in key.lower() or 'fc' in key.lower() or 'head' in key.lower():
                        if 'weight' in key and len(tensor.shape) == 2:
                            model_num_classes = tensor.shape[0]  # Output dimension
                            break
                        elif 'bias' in key and len(tensor.shape) == 1:
                            model_num_classes = tensor.shape[0]
                            break
                
                if model_num_classes is None:
                    if self.verbose:
                        print(f"   ⚠️ Could not detect model's class count, using dataset classes: {self.num_classes}")
                    model_num_classes = self.num_classes
                else:
                    if self.verbose:
                        print(f"   🔍 Detected model classes: {model_num_classes}")
                        print(f"   📊 Dataset classes: {self.num_classes}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Could not analyze checkpoint structure: {e}")
                model_num_classes = self.num_classes
        
        # Handle class mismatch - store info but keep dataset classes for evaluation
        if model_num_classes != self.num_classes:
            if model_num_classes < self.num_classes:
                if self.verbose:
                    print(f"   🎯 Model has fewer classes ({model_num_classes}) than dataset ({self.num_classes})")
                    print(f"   📝 Will evaluate on full dataset but treat out-of-range labels as misclassified")
                    print(f"   ⚠️ Classes {model_num_classes}-{self.num_classes-1} will be automatically wrong")
                
                self.model_num_classes = model_num_classes
                self.class_mismatch = True
                # Keep original class names and count for evaluation
                
                if self.verbose:
                    print(f"   📊 Model can predict: {self.class_names[:model_num_classes]}")
                    print(f"   ❌ Model cannot predict: {self.class_names[model_num_classes:]}")
            else:
                if self.verbose:
                    print(f"   ⚠️ Model has more classes than dataset - this may cause issues")
                self.class_mismatch = False
                self.model_num_classes = model_num_classes
        else:
            self.class_mismatch = False
            self.model_num_classes = model_num_classes
        
        # Initialize model architecture with determined class count
        model, input_size, transform, model_config = initialize_model(
            model_name,
            num_classes=model_num_classes,
            use_pretrained=False,  # We'll load trained weights
            feature_extract=False
        )
        
        # Load checkpoint weights with better error handling for class mismatches
        try:
            checkpoint = torch.load(model_checkpoint, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict_to_load = checkpoint['model_state_dict']
                if self.verbose:
                    print(f"   📊 Loading from checkpoint dict")
                    if 'epoch' in checkpoint:
                        print(f"   📈 Checkpoint from epoch: {checkpoint['epoch']}")
                    if 'best_acc' in checkpoint:
                        print(f"   🎯 Best accuracy: {checkpoint['best_acc']:.4f}")
            else:
                state_dict_to_load = checkpoint
                if self.verbose:
                    print(f"   📊 Loading weights directly")
            
            # Try to load state dict with error handling for mismatched layers
            try:
                model.load_state_dict(state_dict_to_load, strict=True)
                if self.verbose:
                    print(f"   ✅ Loaded model weights successfully")
            except RuntimeError as load_error:
                if "size mismatch" in str(load_error):
                    if self.verbose:
                        print(f"   ⚠️ Size mismatch detected, trying partial loading...")
                        print(f"   🔧 This can happen when config model_num_classes differs from checkpoint")
                    
                    # Try loading with strict=False to skip mismatched layers
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict_to_load, strict=False)
                    
                    if self.verbose:
                        if missing_keys:
                            print(f"   📝 Missing keys: {missing_keys}")
                        if unexpected_keys:
                            print(f"   📝 Unexpected keys: {unexpected_keys}")
                        print(f"   ⚠️ Loaded with partial weights - some layers may be randomly initialized")
                        print(f"   💡 This is expected when model_num_classes in config != checkpoint classes")
                else:
                    # Re-raise if it's not a size mismatch error
                    raise load_error
            
            model.to(self.device)
            model.eval()
            
            self.model = model
            self.transform = transform
            self.model_config = model_config
            
            if self.verbose:
                print(f"   📍 Model loaded and set to eval mode")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model: {e}")
    
    def _filter_dataset_for_model_classes(self, dataset):
        """Filter dataset to only include classes that the model can predict."""
        if not hasattr(self, 'class_mismatch') or not self.class_mismatch:
            return dataset
        
        if self.verbose:
            print(f"   🔍 Filtering dataset for model's {self.model_num_classes} classes...")
        
        # Create filtered dataset
        filtered_indices = []
        filtered_samples = []
        
        for i, (sample_path, label) in enumerate(dataset.samples if hasattr(dataset, 'samples') else enumerate(dataset)):
            if hasattr(dataset, 'samples'):
                # For annotation-based datasets
                if label < self.model_num_classes:
                    filtered_indices.append(i)
                    filtered_samples.append((sample_path, label))
            else:
                # For other dataset types
                try:
                    _, label = dataset[i]
                    if label < self.model_num_classes:
                        filtered_indices.append(i)
                except:
                    continue
        
        if hasattr(dataset, 'samples'):
            # Update samples list for annotation datasets
            original_count = len(dataset.samples)
            dataset.samples = filtered_samples
            
            if self.verbose:
                print(f"   ✂️ Filtered: {original_count} → {len(filtered_samples)} samples")
                print(f"   📊 Kept classes: {[self.class_names[i] for i in range(self.model_num_classes)]}")
        
        return dataset

    def load_dataset(self, split='test', custom_annotation=None, custom_root=None):
        """Load dataset - no filtering, evaluate on full dataset."""
        if self.verbose:
            print(f"\n📂 Loading {split} dataset...")
        
        # Use custom paths if provided
        if custom_annotation and custom_root:
            if self.verbose:
                print(f"   📝 Using custom annotation: {custom_annotation}")
                print(f"   📁 Using custom root: {custom_root}")
            
            dataset = AnnotationDataset(
                custom_annotation,
                custom_root,
                transform=self.transform,
                class_names=self.class_names,  # Use full class names
                class_remapping=self.config.get('class_remapping', {})
            )
            
            if self.verbose:
                print(f"   ✅ Loaded custom dataset: {len(dataset)} samples")
                if hasattr(self, 'class_mismatch') and self.class_mismatch:
                    print(f"   📊 Dataset has {self.num_classes} classes, model can predict {self.model_num_classes}")
            
            return dataset
        
        # Load from configuration
        datasets_config = self.config.get('datasets', [])
        if not datasets_config:
            raise ValueError("❌ No datasets configuration found")
        
        # Use first dataset configuration
        dataset_config = datasets_config[0]
        data_dir = self.config['data_dir']
        
        # Get class remapping config
        class_remapping_config = self.config.get('class_remapping', {})
        
        dataset = None
        
        if dataset_config.get('type', 'annotation').lower() == 'annotation':
            # Try different split annotations
            ann_path_key = f'annotation_{split}'
            ann_root_key = f'annotation_{split}_root'
            
            ann_path = dataset_config.get(ann_path_key)
            if ann_path:
                ann_root = dataset_config.get(
                    ann_root_key, 
                    dataset_config.get('annotation_root', data_dir)
                )
                
                # Resolve path
                if not os.path.isabs(ann_path):
                    full_ann_path = os.path.join(data_dir, ann_path)
                else:
                    full_ann_path = ann_path
                
                if os.path.exists(full_ann_path):
                    dataset = AnnotationDataset(
                        full_ann_path,
                        ann_root,
                        transform=self.transform,
                        class_names=self.class_names,  # Use full class names
                        class_remapping=class_remapping_config
                    )
                    
                    if self.verbose:
                        print(f"   ✅ Loaded {split} annotation dataset: {len(dataset)} samples")
                        if hasattr(self, 'class_mismatch') and self.class_mismatch:
                            print(f"   📊 Dataset has {self.num_classes} classes, model can predict {self.model_num_classes}")
                else:
                    if self.verbose:
                        print(f"   ⚠️ Annotation file not found: {full_ann_path}")
        
        elif dataset_config.get('type', '').lower() == 'imagefolder':
            imgf_root = dataset_config.get('imagefolder_root')
            subdir_key = f'imagefolder_{split}_subdir'
            subdir = dataset_config.get(subdir_key)
            
            if imgf_root and subdir:
                split_dir = os.path.join(imgf_root, subdir)
                if os.path.isdir(split_dir):
                    dataset = ImageFolderWrapper(
                        root=split_dir,
                        transform=self.transform,
                        class_remapping=class_remapping_config
                    )
                    
                    if self.verbose:
                        print(f"   ✅ Loaded {split} ImageFolder dataset: {len(dataset)} samples")
                        if hasattr(self, 'class_mismatch') and self.class_mismatch:
                            print(f"   📊 Dataset has {self.num_classes} classes, model can predict {self.model_num_classes}")
        
        if dataset is None:
            raise ValueError(f"❌ Could not load {split} dataset from configuration")
        
        return dataset
    
    def run_inference(self, dataset, batch_size=32, save_scores=False, scores_output_path=None):
        """Run inference on dataset with class mismatch handling and optional score saving."""
        if self.verbose:
            print(f"\n🔮 Running inference...")
            if hasattr(self, 'class_mismatch') and self.class_mismatch:
                print(f"   ⚠️ Model can only predict {self.model_num_classes} classes out of {self.num_classes}")
                print(f"   📝 Samples with labels >= {self.model_num_classes} will be treated as misclassified")
            if save_scores:
                print(f"   💾 Will save softmax scores to file")
        
        # Create dataloader with Windows-compatible settings
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Windows-safe
            collate_fn=collate_fn_skip_error,
            pin_memory=False
        )
        
        all_predictions = []
        all_true_labels = []
        all_confidences = []
        all_probabilities = []
        all_sample_paths = []
        scores_lines = []  # For saving detailed scores
        
        self.model.eval()
        total_samples = 0
        correct_predictions = 0
        out_of_range_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # Handle different return formats from dataloader
                if len(batch_data) == 2:
                    images, labels = batch_data
                    sample_paths = None
                elif len(batch_data) == 3:
                    images, labels, sample_paths = batch_data
                else:
                    continue
                
                if images is None or labels is None:
                    continue
                
                # Get sample paths for score logging
                if sample_paths is None and hasattr(dataset, 'samples'):
                    # Try to get paths from dataset
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + len(images), len(dataset.samples))
                    sample_paths = [dataset.samples[i][0] for i in range(start_idx, end_idx)]
                elif sample_paths is None:
                    # Generate placeholder paths
                    sample_paths = [f"sample_{total_samples + i}" for i in range(len(images))]
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                # Handle class mismatch - mark out-of-range labels as wrong
                if hasattr(self, 'class_mismatch') and self.class_mismatch:
                    # For labels >= model_num_classes, we'll assign a dummy prediction
                    # that ensures they're counted as wrong
                    out_of_range_mask = labels >= self.model_num_classes
                    out_of_range_count = out_of_range_mask.sum().item()
                    out_of_range_samples += out_of_range_count
                    
                    if out_of_range_count > 0:
                        # For out-of-range samples, assign a prediction that's guaranteed to be wrong
                        # We'll use (true_label + 1) % model_num_classes to ensure mismatch
                        wrong_predictions = (labels[out_of_range_mask] + 1) % self.model_num_classes
                        predicted[out_of_range_mask] = wrong_predictions
                        
                        # Set low confidence for these forced wrong predictions
                        confidences[out_of_range_mask] = 0.01
                
                # Convert to CPU for storage
                batch_predictions = predicted.cpu().numpy()
                batch_labels = labels.cpu().numpy()
                batch_confidences = confidences.cpu().numpy()
                batch_probabilities = probabilities.cpu().numpy()
                
                # Store results
                all_predictions.extend(batch_predictions)
                all_true_labels.extend(batch_labels)
                all_confidences.extend(batch_confidences)
                all_sample_paths.extend(sample_paths)
                
                # For probabilities, we need to handle the dimension mismatch
                if hasattr(self, 'class_mismatch') and self.class_mismatch:
                    # Extend probabilities to match dataset classes
                    extended_probs = np.zeros((batch_probabilities.shape[0], self.num_classes))
                    extended_probs[:, :self.model_num_classes] = batch_probabilities
                    # Leave probabilities for unavailable classes as 0
                    all_probabilities.extend(extended_probs)
                    
                    # For score saving, use extended probabilities
                    if save_scores:
                        for i in range(len(batch_labels)):
                            path = sample_paths[i] if i < len(sample_paths) else f"sample_{total_samples + i}"
                            true_label = batch_labels[i]
                            pred_label = batch_predictions[i]
                            scores = extended_probs[i]
                            
                            # Format: path,true_label,predicted_label,score_0,score_1,...,score_N
                            score_str = ",".join([f"{score:.6f}" for score in scores])
                            line = f"{path},{true_label},{pred_label},{score_str}"
                            scores_lines.append(line)
                else:
                    all_probabilities.extend(batch_probabilities)
                    
                    # For score saving, use original probabilities
                    if save_scores:
                        for i in range(len(batch_labels)):
                            path = sample_paths[i] if i < len(sample_paths) else f"sample_{total_samples + i}"
                            true_label = batch_labels[i]
                            pred_label = batch_predictions[i]
                            scores = batch_probabilities[i]
                            
                            # Format: path,true_label,predicted_label,score_0,score_1,...,score_N
                            score_str = ",".join([f"{score:.6f}" for score in scores])
                            line = f"{path},{true_label},{pred_label},{score_str}"
                            scores_lines.append(line)
                
                # Track accuracy
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                if batch_idx % 20 == 0 and batch_idx > 0:
                    if self.verbose:
                        print(f"   Processed {batch_idx * batch_size} samples...")
        
        # Save scores to file if requested
        if save_scores and scores_lines:
            if scores_output_path is None:
                scores_output_path = "inference_scores.txt"
            
            try:
                with open(scores_output_path, 'w', encoding='utf-8') as f:
                    # Write header
                    class_headers = ",".join([f"score_{name}" for name in self.class_names])
                    header = f"sample_path,true_label,predicted_label,{class_headers}\n"
                    f.write(header)
                    
                    # Write data
                    for line in scores_lines:
                        f.write(line + "\n")
                
                if self.verbose:
                    print(f"   💾 Softmax scores saved to: {scores_output_path}")
                    print(f"   📊 Saved scores for {len(scores_lines)} samples")
                    
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Error saving scores file: {e}")
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        if self.verbose:
            print(f"   ✅ Inference completed: {len(all_predictions)} predictions")
            print(f"   📊 Overall Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
            if hasattr(self, 'class_mismatch') and self.class_mismatch and out_of_range_samples > 0:
                print(f"   ⚠️ Out-of-range samples: {out_of_range_samples} (automatically wrong)")
                in_range_samples = total_samples - out_of_range_samples
                in_range_correct = correct_predictions  # All correct are in-range by definition
                in_range_accuracy = in_range_correct / in_range_samples if in_range_samples > 0 else 0
                print(f"   📈 In-range accuracy: {in_range_accuracy:.4f} ({in_range_correct}/{in_range_samples})")
        
        return {
            'predictions': np.array(all_predictions),
            'true_labels': np.array(all_true_labels),
            'confidences': np.array(all_confidences),
            'probabilities': np.array(all_probabilities),
            'sample_paths': all_sample_paths,
            'accuracy': accuracy,
            'total_samples': total_samples,
            'out_of_range_samples': out_of_range_samples if hasattr(self, 'class_mismatch') and self.class_mismatch else 0,
            'scores_saved': save_scores,
            'scores_file': scores_output_path if save_scores else None
        }

    def generate_classification_report(self, results, save_dir=None):
        """Generate comprehensive classification report using existing evaluation functions."""
        predictions = results['predictions']
        true_labels = results['true_labels']
        confidences = results['confidences']
        
        if self.verbose:
            print(f"\n📊 Generating classification report...")
            if hasattr(self, 'class_mismatch') and self.class_mismatch:
                print(f"   📋 Evaluating on all {self.num_classes} dataset classes")
                print(f"   🤖 Model can predict: {self.model_num_classes} classes")
                if results.get('out_of_range_samples', 0) > 0:
                    print(f"   ⚠️ Out-of-range samples: {results['out_of_range_samples']} (counted as wrong)")
        
        # Basic metrics - evaluate on all dataset classes
        accuracy = accuracy_score(true_labels, predictions)
        
        # Per-class metrics - use all dataset classes
        valid_labels = list(range(self.num_classes))
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=valid_labels, zero_division=0
        )
        
        # Macro/weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Additional metrics
        kappa = cohen_kappa_score(true_labels, predictions)
        mcc = matthews_corrcoef(true_labels, predictions)
        
        # Create report dictionary with comprehensive info
        report = {
            'overall_metrics': {
                'accuracy': accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1,
                'cohen_kappa': kappa,
                'matthews_corrcoef': mcc
            },
            'per_class_metrics': {},
            'confidence_stats': {
                'mean_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences)
            },
            'class_info': {
                'dataset_classes': self.num_classes,
                'model_classes': getattr(self, 'model_num_classes', self.num_classes),
                'class_names': self.class_names,
                'class_mismatch': getattr(self, 'class_mismatch', False),
                'out_of_range_samples': results.get('out_of_range_samples', 0),
                'total_samples': results.get('total_samples', len(predictions))
            }
        }
        
        # Per-class details - include all classes
        for i, class_name in enumerate(self.class_names):
            if i < len(precision):
                report['per_class_metrics'][class_name] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': int(support[i]),
                    'can_predict': i < getattr(self, 'model_num_classes', self.num_classes)
                }
        
        # Print summary
        if self.verbose:
            print(f"   📈 Results Summary:")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      Macro F1: {macro_f1:.4f}")
            print(f"      Weighted F1: {weighted_f1:.4f}")
            print(f"      Cohen's Kappa: {kappa:.4f}")
            print(f"      Mean Confidence: {report['confidence_stats']['mean_confidence']:.4f}")
            
            if hasattr(self, 'class_mismatch') and self.class_mismatch:
                print(f"      📊 Evaluated on all {self.num_classes} classes (model supports {self.model_num_classes})")
                
                # Show per-class breakdown
                print(f"      📋 Class-wise performance:")
                for i, class_name in enumerate(self.class_names):
                    metrics = report['per_class_metrics'][class_name]
                    status = "✅" if metrics['can_predict'] else "❌"
                    print(f"         {status} {class_name}: F1={metrics['f1']:.3f}, Support={metrics['support']}")
        
        # Use the comprehensive evaluation function to generate reports and confusion matrices
        if save_dir:
            self._save_comprehensive_report(report, true_labels, predictions, save_dir)
        
        return report

    def _save_comprehensive_report(self, report, y_true, y_pred, save_dir):
        """Save comprehensive classification report using existing evaluation functions."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\n💾 Saving comprehensive results to: {save_dir}")
        
        # 1. Use the existing report_classification function for detailed reports and confusion matrices
        save_path_base = str(save_dir / "classification_evaluation")
        
        if self.verbose:
            print(f"   🎯 Generating detailed classification report and confusion matrices...")
        
        # Call the comprehensive evaluation function
        report_classification(
            y_true=y_true.tolist(),
            y_pred=y_pred.tolist(), 
            class_names=self.class_names,
            save_path_base=save_path_base
        )
        
        # 2. Save our custom summary with class mismatch info
        self._save_text_summary(report, save_dir / "inference_summary.txt")
        
        # 3. Save per-class metrics as CSV
        per_class_df = pd.DataFrame(report['per_class_metrics']).transpose()
        per_class_df.to_csv(save_dir / "per_class_metrics.csv")
        
        # 4. Save confidence statistics
        confidence_df = pd.DataFrame([report['confidence_stats']])
        confidence_df.to_csv(save_dir / "confidence_stats.csv", index=False)
        
        # 5. Save class info and mismatch details
        class_info_df = pd.DataFrame([report['class_info']])
        class_info_df.to_csv(save_dir / "class_info.csv", index=False)
        
        if self.verbose:
            print(f"   ✅ Comprehensive results saved successfully!")
            print(f"      📊 Classification report: {save_path_base}_report.txt")
            print(f"      📈 Raw confusion matrix: {save_path_base}_cm_raw.png")
            print(f"      📉 Normalized confusion matrices: {save_path_base}_cm_norm_*.png")
            print(f"      📋 Summary: inference_summary.txt")
            print(f"      📊 Metrics CSV files: per_class_metrics.csv, confidence_stats.csv, class_info.csv")

    def _save_text_summary(self, report, save_path):
        """Save text summary report with full dataset evaluation info."""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("PLASMODIUM CLASSIFICATION INFERENCE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Configuration: {self.config_path}\n")
            f.write(f"Dataset Classes ({self.num_classes}): {self.class_names}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Class mismatch info
            class_info = report.get('class_info', {})
            if class_info.get('class_mismatch', False):
                f.write(f"\nCLASS MISMATCH HANDLING:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Dataset classes: {class_info['dataset_classes']}\n")
                f.write(f"Model classes: {class_info['model_classes']}\n")
                f.write(f"Evaluation: Full dataset ({class_info['dataset_classes']} classes)\n")
                f.write(f"Out-of-range samples: {class_info['out_of_range_samples']}/{class_info['total_samples']}\n")
                f.write(f"Model can predict: {self.class_names[:class_info['model_classes']]}\n")
                f.write(f"Model cannot predict: {self.class_names[class_info['model_classes']:]}\n")
            
            f.write("\nOVERALL METRICS:\n")
            f.write("-" * 20 + "\n")
            for metric, value in report['overall_metrics'].items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nPER-CLASS PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            for class_name, metrics in report['per_class_metrics'].items():
                predictable = "✅" if metrics.get('can_predict', True) else "❌"
                f.write(f"\n{predictable} {class_name}:\n")
                f.write(f"  precision: {metrics['precision']:.4f}\n")
                f.write(f"  recall: {metrics['recall']:.4f}\n")
                f.write(f"  f1: {metrics['f1']:.4f}\n")
                f.write(f"  support: {metrics['support']}\n")
            
            f.write("\nCONFIDENCE STATISTICS:\n")
            f.write("-" * 20 + "\n")
            for stat, value in report['confidence_stats'].items():
                f.write(f"{stat}: {value:.4f}\n")
    
    def run_phase2_evaluation(self, scores_file_path, save_dir=None):
        """
        Phase 2 evaluation: Load saved scores and evaluate using second-highest score as prediction.
        
        Args:
            scores_file_path (str): Path to the saved scores file
            save_dir (str): Directory to save Phase 2 results
            
        Returns:
            dict: Phase 2 evaluation results
        """
        if self.verbose:
            print(f"\n🔬 Starting Phase 2 Evaluation...")
            print(f"   📂 Loading scores from: {scores_file_path}")
            print(f"   🎯 Strategy: Use second-highest score as prediction")
        
        if not os.path.exists(scores_file_path):
            raise FileNotFoundError(f"❌ Scores file not found: {scores_file_path}")
        
        # Load scores file
        try:
            with open(scores_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 2:  # Header + at least one data line
                raise ValueError("❌ Scores file appears to be empty or corrupted")
            
            # Parse header
            header = lines[0].strip().split(',')
            expected_score_cols = len(self.class_names)
            
            if len(header) < 3 + expected_score_cols:
                raise ValueError(f"❌ Invalid header format. Expected at least {3 + expected_score_cols} columns")
            
            if self.verbose:
                print(f"   📋 Header: {header}")
                print(f"   📊 Expected {expected_score_cols} score columns for {len(self.class_names)} classes")
            
            # Parse data lines
            sample_paths = []
            true_labels = []
            original_predictions = []
            phase2_predictions = []
            all_scores = []
            
            for line_idx, line in enumerate(lines[1:], 1):
                parts = line.strip().split(',')
                
                if len(parts) < 3 + expected_score_cols:
                    if self.verbose:
                        print(f"   ⚠️ Skipping malformed line {line_idx}: insufficient columns")
                    continue
                
                try:
                    sample_path = parts[0]
                    true_label = int(parts[1])
                    original_pred = int(parts[2])
                    scores = [float(parts[3 + i]) for i in range(expected_score_cols)]
                    

                    score_indices = list(range(len(scores)))
                    score_indices.sort(key=lambda x: scores[x], reverse=True)
                    
                    if len(score_indices) >= 2:
                        if score_diff_fn(scores[score_indices[0]], scores[score_indices[1]]):
                            phase2_pred = 6
                        else:
                            phase2_pred = score_indices[0]
                    else:
                        phase2_pred = score_indices[0]  # Fallback to highest if only one class
        
                    sample_paths.append(sample_path)
                    true_labels.append(true_label)
                    original_predictions.append(original_pred)
                    phase2_predictions.append(phase2_pred)
                    all_scores.append(scores)
                    
                except (ValueError, IndexError) as e:
                    if self.verbose:
                        print(f"   ⚠️ Error parsing line {line_idx}: {e}")
                    continue
            
            if not true_labels:
                raise ValueError("❌ No valid data lines found in scores file")
            
            if self.verbose:
                print(f"   ✅ Loaded {len(true_labels)} samples successfully")
                print(f"   📊 Original accuracy: {accuracy_score(true_labels, original_predictions):.4f}")
            
        except Exception as e:
            raise RuntimeError(f"❌ Error loading scores file: {e}")
        
        # Calculate Phase 2 metrics
        phase2_accuracy = accuracy_score(true_labels, phase2_predictions)
        
        # Per-class metrics for Phase 2
        valid_labels = list(range(self.num_classes))
        p2_precision, p2_recall, p2_f1, p2_support = precision_recall_fscore_support(
            true_labels, phase2_predictions, average=None, labels=valid_labels, zero_division=0
        )
        
        # Macro/weighted averages for Phase 2
        p2_macro_precision, p2_macro_recall, p2_macro_f1, _ = precision_recall_fscore_support(
            true_labels, phase2_predictions, average='macro', zero_division=0
        )
        p2_weighted_precision, p2_weighted_recall, p2_weighted_f1, _ = precision_recall_fscore_support(
            true_labels, phase2_predictions, average='weighted', zero_division=0
        )
        
        # Additional metrics for Phase 2
        p2_kappa = cohen_kappa_score(true_labels, phase2_predictions)
        p2_mcc = matthews_corrcoef(true_labels, phase2_predictions)
        
        # Calculate confidence statistics for second-highest scores
        second_highest_confidences = []
        for scores in all_scores:
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) >= 2:
                second_highest_confidences.append(sorted_scores[1])
            else:
                second_highest_confidences.append(sorted_scores[0])
        
        # Create Phase 2 report
        phase2_report = {
            'strategy': 'second_highest_score',
            'overall_metrics': {
                'accuracy': phase2_accuracy,
                'macro_precision': p2_macro_precision,
                'macro_recall': p2_macro_recall,
                'macro_f1': p2_macro_f1,
                'weighted_precision': p2_weighted_precision,
                'weighted_recall': p2_weighted_recall,
                'weighted_f1': p2_weighted_f1,
                'cohen_kappa': p2_kappa,
                'matthews_corrcoef': p2_mcc
            },
            'per_class_metrics': {},
            'confidence_stats': {
                'mean_confidence': np.mean(second_highest_confidences),
                'std_confidence': np.std(second_highest_confidences),
                'min_confidence': np.min(second_highest_confidences),
                'max_confidence': np.max(second_highest_confidences)
            },
            'comparison_with_phase1': {
                'original_accuracy': accuracy_score(true_labels, original_predictions),
                'phase2_accuracy': phase2_accuracy,
                'accuracy_difference': phase2_accuracy - accuracy_score(true_labels, original_predictions),
                'total_samples': len(true_labels)
            }
        }
        
        # Per-class details for Phase 2
        for i, class_name in enumerate(self.class_names):
            if i < len(p2_precision):
                phase2_report['per_class_metrics'][class_name] = {
                    'precision': p2_precision[i],
                    'recall': p2_recall[i],
                    'f1': p2_f1[i],
                    'support': int(p2_support[i])
                }
        
        # Print Phase 2 summary
        if self.verbose:
            print(f"\n   📈 Phase 2 Results Summary:")
            print(f"      Strategy: Second-highest score prediction")
            print(f"      Phase 2 Accuracy: {phase2_accuracy:.4f}")
            print(f"      Original Accuracy: {phase2_report['comparison_with_phase1']['original_accuracy']:.4f}")
            print(f"      Accuracy Change: {phase2_report['comparison_with_phase1']['accuracy_difference']:+.4f}")
            print(f"      Phase 2 Macro F1: {p2_macro_f1:.4f}")
            print(f"      Phase 2 Weighted F1: {p2_weighted_f1:.4f}")
            print(f"      Mean 2nd-highest Confidence: {phase2_report['confidence_stats']['mean_confidence']:.4f}")
            
            # Show per-class Phase 2 performance
            print(f"      📋 Phase 2 Class-wise performance:")
            for i, class_name in enumerate(self.class_names):
                if class_name in phase2_report['per_class_metrics']:
                    metrics = phase2_report['per_class_metrics'][class_name]
                    print(f"         🔬 {class_name}: F1={metrics['f1']:.3f}, Support={metrics['support']}")
        
        # Save Phase 2 results
        if save_dir:
            self._save_phase2_report(phase2_report, true_labels, phase2_predictions, original_predictions, save_dir)
        
        return {
            'true_labels': true_labels,
            'original_predictions': original_predictions,
            'phase2_predictions': phase2_predictions,
            'all_scores': all_scores,
            'sample_paths': sample_paths,
            'phase2_report': phase2_report
        }

    def _save_phase2_report(self, phase2_report, y_true, y_pred_phase2, y_pred_original, save_dir):
        """Save Phase 2 evaluation results."""
        save_dir = Path(save_dir)
        phase2_dir = save_dir / "phase2_evaluation"
        phase2_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\n   💾 Saving Phase 2 results to: {phase2_dir}")
        
        # 1. Use report_classification for Phase 2 confusion matrices
        phase2_base_path = str(phase2_dir / "phase2_classification_evaluation")
        
        if self.verbose:
            print(f"      🎯 Generating Phase 2 classification report and confusion matrices...")
        
        report_classification(
            y_true=y_true,
            y_pred=y_pred_phase2,
            class_names=self.class_names,
            save_path_base=phase2_base_path
        )
        
        # 2. Save Phase 2 summary
        self._save_phase2_text_summary(phase2_report, phase2_dir / "phase2_summary.txt")
        
        # 3. Save Phase 2 per-class metrics
        phase2_per_class_df = pd.DataFrame(phase2_report['per_class_metrics']).transpose()
        phase2_per_class_df.to_csv(phase2_dir / "phase2_per_class_metrics.csv")
        
        # 4. Save comparison metrics
        comparison_df = pd.DataFrame([phase2_report['comparison_with_phase1']])
        comparison_df.to_csv(phase2_dir / "phase1_vs_phase2_comparison.csv", index=False)
        
        # 5. Save detailed prediction comparison
        comparison_data = {
            'sample_index': list(range(len(y_true))),
            'true_label': y_true,
            'phase1_prediction': y_pred_original,
            'phase2_prediction': y_pred_phase2,
            'phase1_correct': [t == p1 for t, p1 in zip(y_true, y_pred_original)],
            'phase2_correct': [t == p2 for t, p2 in zip(y_true, y_pred_phase2)],
            'prediction_changed': [p1 != p2 for p1, p2 in zip(y_pred_original, y_pred_phase2)],
            'improvement': [(t == p2 and t != p1) for t, p1, p2 in zip(y_true, y_pred_original, y_pred_phase2)],
            'degradation': [(t == p1 and t != p2) for t, p1, p2 in zip(y_true, y_pred_original, y_pred_phase2)]
        }
        comparison_detailed_df = pd.DataFrame(comparison_data)
        comparison_detailed_df.to_csv(phase2_dir / "detailed_prediction_comparison.csv", index=False)
        
        if self.verbose:
            print(f"      ✅ Phase 2 results saved successfully!")
            print(f"         📊 Phase 2 classification report: {phase2_base_path}_report.txt")
            print(f"         📈 Phase 2 confusion matrices: {phase2_base_path}_cm_*.png")
            print(f"         📋 Phase 2 summary: phase2_summary.txt")
            print(f"         📊 Detailed comparison: detailed_prediction_comparison.csv")

    def _save_phase2_text_summary(self, phase2_report, save_path):
        """Save Phase 2 text summary."""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("PHASE 2 EVALUATION - SECOND-HIGHEST SCORE PREDICTION\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Strategy: Use second-highest softmax score as prediction class\n")
            f.write(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Classes ({self.num_classes}): {self.class_names}\n\n")
            
            # Comparison with Phase 1
            comp = phase2_report['comparison_with_phase1']
            f.write("PHASE 1 vs PHASE 2 COMPARISON:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total samples: {comp['total_samples']}\n")
            f.write(f"Phase 1 accuracy (highest score): {comp['original_accuracy']:.4f}\n")
            f.write(f"Phase 2 accuracy (2nd-highest score): {comp['phase2_accuracy']:.4f}\n")
            f.write(f"Accuracy difference: {comp['accuracy_difference']:+.4f}\n")
            
            if comp['accuracy_difference'] > 0:
                f.write("Result: Phase 2 performs BETTER than Phase 1! 🎉\n")
            elif comp['accuracy_difference'] < 0:
                f.write("Result: Phase 2 performs worse than Phase 1.\n")
            else:
                f.write("Result: Phase 2 performs the same as Phase 1.\n")
            
            # Phase 2 metrics
            f.write("\nPHASE 2 OVERALL METRICS:\n")
            f.write("-" * 25 + "\n")
            for metric, value in phase2_report['overall_metrics'].items():
                f.write(f"{metric}: {value:.4f}\n")
            
            # Phase 2 per-class performance
            f.write("\nPHASE 2 PER-CLASS PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for class_name, metrics in phase2_report['per_class_metrics'].items():
                f.write(f"\n🔬 {class_name}:\n")
                f.write(f"  precision: {metrics['precision']:.4f}\n")
                f.write(f"  recall: {metrics['recall']:.4f}\n")
                f.write(f"  f1: {metrics['f1']:.4f}\n")
                f.write(f"  support: {metrics['support']}\n")
            
            # Confidence statistics
            f.write("\nPHASE 2 CONFIDENCE STATISTICS (2nd-highest scores):\n")
            f.write("-" * 45 + "\n")
            for stat, value in phase2_report['confidence_stats'].items():
                f.write(f"{stat}: {value:.4f}\n")


def run_simple_inference(
    model_checkpoint=None,
    model_name=None,
    model_num_classes=None,
    config_path='config_local.yaml',
    split='test',
    custom_annotation=None,
    custom_root=None,
    output_dir=None,
    batch_size=32,
    save_scores=False,
    scores_filename=None,
    run_phase2=False,
    verbose=True
):
    """
    🚀 Run simple inference with direct model specification.
    
    Args:
        model_checkpoint (str): Direct path to model checkpoint file
        model_name (str): Model name for timm initialization (e.g., 'deit_small_patch16_224')
        model_num_classes (int): Number of classes for model initialization (overrides auto-detection)
        config_path (str): Path to configuration file
        split (str): Dataset split to evaluate ('test', 'val', 'train')
        custom_annotation (str): Path to custom annotation file
        custom_root (str): Root directory for custom annotation
        output_dir (str): Output directory for results
        batch_size (int): Batch size for inference
        save_scores (bool): Whether to save softmax scores to text file
        scores_filename (str): Custom filename for scores file (optional)
        run_phase2 (bool): Whether to run Phase 2 evaluation (requires save_scores=True)
        verbose (bool): Print detailed progress
    
    Returns:
        dict: Complete inference results and evaluation report
    """
    
    try:
        # Initialize inference system
        inference = SimpleInference(config_path=config_path, verbose=verbose)
        
        # Override model_num_classes in config if provided
        if model_num_classes is not None:
            inference.config['model_num_classes'] = model_num_classes
            if verbose:
                print(f"🔧 Overriding model classes to: {model_num_classes}")
        
        # Validate Phase 2 requirements
        if run_phase2 and not save_scores:
            if verbose:
                print(f"⚠️ Phase 2 evaluation requires save_scores=True. Enabling score saving.")
            save_scores = True
        
        # Select and load model
        selected_checkpoint = inference.select_model_checkpoint(model_checkpoint=model_checkpoint)
        model = inference.load_model(selected_checkpoint, model_name=model_name)
        
        # Load dataset
        dataset = inference.load_dataset(
            split=split,
            custom_annotation=custom_annotation,
            custom_root=custom_root
        )
        
        # Prepare scores output path
        scores_output_path = None
        if save_scores:
            if output_dir is None:
                checkpoint_dir = Path(selected_checkpoint).parent
                output_dir = checkpoint_dir / "simple_inference_results"
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if scores_filename is None:
                scores_filename = f"inference_scores_{split}.txt"
            
            scores_output_path = str(output_dir / scores_filename)
        
        # Run inference with score saving
        results = inference.run_inference(
            dataset, 
            batch_size=batch_size,
            save_scores=save_scores,
            scores_output_path=scores_output_path
        )
        
        # Generate evaluation report
        if output_dir is None:
            checkpoint_dir = Path(selected_checkpoint).parent
            output_dir = checkpoint_dir / "simple_inference_results"
        
        report = inference.generate_classification_report(results, save_dir=output_dir)
        
        # Run Phase 2 evaluation if requested
        phase2_results = None
        if run_phase2 and results.get('scores_file'):
            if verbose:
                print(f"\n🔬 Starting Phase 2 evaluation...")
            
            try:
                phase2_results = inference.run_phase2_evaluation(
                    scores_file_path=results['scores_file'],
                    save_dir=output_dir
                )
                
                if verbose:
                    print(f"✅ Phase 2 evaluation completed successfully!")
                    
            except Exception as e:
                if verbose:
                    print(f"❌ Phase 2 evaluation failed: {e}")
                phase2_results = None
        
        # Combine results
        final_results = {
            'model_checkpoint': selected_checkpoint,
            'model_name': model_name,
            'dataset_size': len(dataset),
            'class_names': inference.class_names,
            'inference_results': results,
            'evaluation_report': report,
            'output_directory': str(output_dir),
            'scores_saved': save_scores,
            'scores_file': results.get('scores_file') if save_scores else None,
            'phase2_enabled': run_phase2,
            'phase2_results': phase2_results
        }
        
        if verbose:
            print(f"\n🎉 Simple inference completed successfully!")
            print(f"📁 Results saved to: {output_dir}")
            print(f"🤖 Model: {model_name or 'auto-detected'}")
            print(f"📊 Dataset size: {len(dataset)} samples")
            print(f"🎯 Phase 1 accuracy: {results['accuracy']:.4f}")
            if save_scores and results.get('scores_file'):
                print(f"💾 Softmax scores saved to: {results['scores_file']}")
            if phase2_results:
                p2_acc = phase2_results['phase2_report']['overall_metrics']['accuracy']
                print(f"🔬 Phase 2 accuracy: {p2_acc:.4f}")
                diff = p2_acc - results['accuracy']
                print(f"📊 Phase 2 vs Phase 1: {diff:+.4f}")
        
        return final_results
        
    except Exception as e:
        if verbose:
            print(f"❌ Error during simple inference: {e}")
            import traceback
            traceback.print_exc()
        raise e


def main():
    """Main function for standalone script execution."""
    print("🎯 PlasmodiumClassification Simple Inference")
    print("=" * 50)
    model_checkpoint = r"X:\datn\6cls_results\mobilenetv4_hybrid_medium.ix_e550_r384_in1\PlasmodiumClassification-1\results_kaggle\mobilenetv4_hybrid_medium.ix_e550_r384_in1k\mobilenetv4_hybrid_medium.ix_e550_r384_in1k_best.pth"
    model_name = 'mobilenetv4_hybrid_medium.ix_e550_r384_in1k'
    model_num_classes = 6
    
    # Example usage - specify direct model name, checkpoint path, and class count
    results = run_simple_inference(
        model_name=model_name,  # Direct timm model name
        model_checkpoint=model_checkpoint,  # Direct path
        model_num_classes=model_num_classes,  # Explicitly specify model class count
        split='test',
        batch_size=16,
        save_scores=True,  # 💾 Enable softmax score saving
        scores_filename="test_scores_6cls_vs_7cls.txt",  # Custom filename
        run_phase2=True,  # 🔬 Enable Phase 2 evaluation
        verbose=True
    )
    
    if results:
        print(f"\n✨ Inference completed successfully!")
        print(f"📊 Final metrics:")
        metrics = results['evaluation_report']['overall_metrics']
        print(f"   Phase 1 Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Phase 1 Macro F1: {metrics['macro_f1']:.4f}")
        print(f"   Phase 1 Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        
        if results.get('scores_saved'):
            print(f"   💾 Scores file: {results['scores_file']}")
            
        if results.get('phase2_results'):
            p2_metrics = results['phase2_results']['phase2_report']['overall_metrics']
            print(f"   🔬 Phase 2 Accuracy: {p2_metrics['accuracy']:.4f}")
            print(f"   🔬 Phase 2 Macro F1: {p2_metrics['macro_f1']:.4f}")
            print(f"   📊 Accuracy Improvement: {p2_metrics['accuracy'] - metrics['accuracy']:+.4f}")


if __name__ == "__main__":
    # Example usage patterns:
    
    # 1. Direct model specification (recommended)
    # results = run_simple_inference(
    #     model_name="deit_small_patch16_224",
    #     model_checkpoint="path/to/your/model_best.pth",
    #     verbose=True
    # )
    
    # 2. Auto model selection (fallback)
    # results = run_simple_inference(verbose=True)
    
    # 3. Custom dataset with direct model
    # results = run_simple_inference(
    #     model_name="resnet50",
    #     model_checkpoint="path/to/resnet50_best.pth",
    #     custom_annotation="path/to/custom_annotation.txt",
    #     custom_root="path/to/custom/images",
    #     output_dir="custom_results",
    #     verbose=True
    # )
    
    # Run main example
    main()
