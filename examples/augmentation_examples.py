"""
Example Integration of Advanced Augmentation
==========================================

This script demonstrates how to integrate the new augmentation module
into the existing PlasmodiumClassification project.

Usage examples:
1. Basic augmentation strategies
2. Custom augmentation configuration
3. Integration with main training pipeline
4. MixUp/CutMix during training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import existing modules
from src.augment import (
    TimmAugmentationStrategy, 
    MixupCutmixWrapper, 
    RandAugmentTransform,
    create_augmentation_strategy,
    get_timm_transform,
    test_augmentations
)
from src.data_loader import AnnotationDataset, ImageFolderWrapper
from src.model_initializer import initialize_model


def example_basic_usage():
    """Demonstrate basic usage of augmentation strategies."""
    print("🔄 Example 1: Basic Augmentation Strategies")
    print("=" * 50)
    
    # Create different augmentation strategies
    strategies = ['minimal', 'light', 'medium', 'heavy', 'extreme']
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} Strategy ---")
        
        # Create augmentation strategy
        aug_strategy = TimmAugmentationStrategy(
            strategy=strategy,
            input_size=224,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        # Get transforms
        train_transform = aug_strategy.get_train_transform()
        eval_transform = aug_strategy.get_eval_transform()
        
        print(f"✅ Train transform: {len(train_transform.transforms)} steps")
        print(f"✅ Eval transform: {len(eval_transform.transforms)} steps")


def example_config_integration():
    """Demonstrate integration with config files."""
    print("\n🔄 Example 2: Config-Based Augmentation")
    print("=" * 50)
    
    # Example config (similar to what would be in config.yaml)
    config = {
        'augmentation': {
            'enabled': True,
            'strategy': 'medium',
            'auto_augment_policy': 'original',
            'randaugment_magnitude': 9,
            'randaugment_num_ops': 2,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0,
            'random_erase_prob': 0.25,
            'use_timm_auto_augment': True,
            'color_jitter': 0.4,
            'scale_range': [0.7, 1.0],
            'ratio_range': [0.8, 1.2]
        }
    }
    
    # Example model config
    model_config = {
        'input_size': 384,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'interpolation': 'bicubic'
    }
    
    # Create augmentation strategy from config
    aug_strategy = create_augmentation_strategy(config, model_config)
    
    train_transform = aug_strategy.get_train_transform()
    eval_transform = aug_strategy.get_eval_transform()
    
    print(f"✅ Created augmentation strategy: {aug_strategy.strategy}")
    print(f"✅ Input size: {aug_strategy.input_size}")
    print(f"✅ Interpolation: {aug_strategy.interpolation}")


def example_dataset_integration():
    """Demonstrate how to use augmentation with datasets."""
    print("\n🔄 Example 3: Dataset Integration")
    print("=" * 50)
    
    # Create augmentation strategy
    aug_strategy = TimmAugmentationStrategy(
        strategy='light',
        input_size=224
    )
    
    # Get transforms
    train_transform = aug_strategy.get_train_transform()
    eval_transform = aug_strategy.get_eval_transform()
    
    print("✅ Transforms created for dataset integration")
    print("   Use these transforms when creating datasets:")
    print("   train_dataset = AnnotationDataset(..., transform=train_transform)")
    print("   val_dataset = AnnotationDataset(..., transform=eval_transform)")
    
    # Example of how this would be used in practice:
    """
    # In your main training script:
    config = load_config('config.yaml')
    model, input_size, _, model_config = initialize_model(model_name, num_classes)
    
    # Create augmentation strategy
    aug_strategy = create_augmentation_strategy(config, model_config)
    train_transform = aug_strategy.get_train_transform()
    eval_transform = aug_strategy.get_eval_transform()
    
    # Create datasets with augmentation
    train_dataset = AnnotationDataset(
        annotation_file='train.txt',
        root_dir='images/',
        transform=train_transform,
        class_names=class_names
    )
    
    val_dataset = AnnotationDataset(
        annotation_file='val.txt', 
        root_dir='images/',
        transform=eval_transform,
        class_names=class_names
    )
    """


def example_mixup_cutmix():
    """Demonstrate MixUp/CutMix usage during training."""
    print("\n🔄 Example 4: MixUp/CutMix Integration")
    print("=" * 50)
    
    # Create MixUp/CutMix wrapper
    mixup_cutmix = MixupCutmixWrapper(
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        num_classes=6,  # For your Plasmodium classification
        label_smoothing=0.1
    )
    
    print(f"✅ MixUp/CutMix enabled: {mixup_cutmix.is_enabled()}")
    
    if mixup_cutmix.is_enabled():
        print("   Integration example for training loop:")
        print("""
        # In your training loop:
        for inputs, labels in dataloader:
            # Apply MixUp/CutMix
            inputs, labels = mixup_cutmix(inputs, labels)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss (labels might be mixed)
            loss = criterion(outputs, labels)
            
            # Rest of training step...
        """)


def example_timm_specific():
    """Demonstrate timm-specific features."""
    print("\n🔄 Example 5: timm-Specific Features")
    print("=" * 50)
    
    # Try to get timm transform for a specific model
    model_name = 'mobilenetv4_hybrid_medium.ix_e550_r384_in1k'
    
    # Get timm's recommended transform
    timm_transform_train = get_timm_transform(
        model_name=model_name,
        input_size=384,
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1',
        color_jitter=0.4,
        re_prob=0.25
    )
    
    timm_transform_eval = get_timm_transform(
        model_name=model_name,
        input_size=384,
        is_training=False
    )
    
    if timm_transform_train and timm_transform_eval:
        print(f"✅ Got timm transforms for {model_name}")
        print(f"   Train: {len(timm_transform_train.transforms) if hasattr(timm_transform_train, 'transforms') else 'Complex transform'}")
        print(f"   Eval: {len(timm_transform_eval.transforms) if hasattr(timm_transform_eval, 'transforms') else 'Complex transform'}")
    else:
        print("⚠️ Could not get timm transforms (timm might not be available)")


def example_custom_randaugment():
    """Demonstrate custom RandAugment implementation."""
    print("\n🔄 Example 6: Custom RandAugment")
    print("=" * 50)
    
    # Create custom RandAugment
    randaugment = RandAugmentTransform(
        num_ops=2,
        magnitude=9,
        num_magnitude_bins=31
    )
    
    # Use in a transform pipeline
    custom_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        randaugment,  # Apply RandAugment
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    print("✅ Custom RandAugment transform created")
    print(f"   Operations per image: {randaugment.num_ops}")
    print(f"   Magnitude: {randaugment.magnitude}")


def example_integration_with_main():
    """Show how to integrate with the main training script."""
    print("\n🔄 Example 7: Integration with main.py")
    print("=" * 50)
    
    print("""
To integrate with your existing main.py, make these changes:

1. Add import at the top:
   from src.augment import create_augmentation_strategy, MixupCutmixWrapper

2. Replace the current transform creation with:
   
   # After initializing the model
   model, input_size, model_specific_transform, model_config = initialize_model(...)
   
   # Create augmentation strategy
   if config.get('augmentation', {}).get('enabled', False):
       aug_strategy = create_augmentation_strategy(config, model_config)
       transform_train = aug_strategy.get_train_transform()
       transform_eval = aug_strategy.get_eval_transform()
       print("✅ Using advanced augmentation strategy")
   else:
       # Use existing transform logic
       transform_train = model_specific_transform or default_transform_train
       transform_eval = model_specific_transform or default_transform_eval

3. For MixUp/CutMix, add to training loop:
   
   # After creating dataloaders
   mixup_cutmix = None
   if config.get('augmentation', {}).get('mixup_alpha', 0) > 0:
       mixup_cutmix = MixupCutmixWrapper(
           mixup_alpha=config['augmentation']['mixup_alpha'],
           cutmix_alpha=config['augmentation']['cutmix_alpha'],
           num_classes=len(final_class_names),
           label_smoothing=config['augmentation'].get('label_smoothing', 0.0)
       )
   
   # Pass to training function
   train_model(..., mixup_cutmix=mixup_cutmix)

4. Update training function to use MixUp/CutMix:
   
   # In training loop, after loading batch
   if mixup_cutmix and mixup_cutmix.is_enabled():
       inputs, labels = mixup_cutmix(inputs, labels)
""")


def main():
    """Run all examples."""
    print("🚀 Advanced Augmentation Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_config_integration()
    example_dataset_integration()
    example_mixup_cutmix()
    example_timm_specific()
    example_custom_randaugment()
    example_integration_with_main()
    
    # Run test function
    print("\n🧪 Running augmentation tests...")
    print("=" * 50)
    test_augmentations()
    
    print("\n✅ All examples completed!")
    print("\nNext steps:")
    print("1. Update your config.yaml with augmentation settings")
    print("2. Modify main.py to use the new augmentation module")
    print("3. Experiment with different strategies to find what works best")
    print("4. Consider using MixUp/CutMix for challenging datasets")


if __name__ == "__main__":
    main()
