"""
Integration Helper for Advanced Augmentation
==========================================

This script provides ready-to-use code snippets that can be directly
integrated into the existing main.py file to enable advanced augmentation.

Copy and paste the relevant sections into your main.py file.
"""

# =============================================================================
# 1. IMPORTS TO ADD AT THE TOP OF main.py
# =============================================================================

"""
Add these imports after the existing imports in main.py:
"""

IMPORTS_TO_ADD = '''
# Advanced augmentation imports (ADD THESE)
from src.augment import (
    create_augmentation_strategy,
    MixupCutmixWrapper,
    get_timm_transform
)
'''

# =============================================================================
# 2. TRANSFORM CREATION REPLACEMENT
# =============================================================================

"""
Replace the current transform creation logic (around line 170-185) with this:
"""

TRANSFORM_CREATION_CODE = '''
                # --- Determine Transform to Use (ENHANCED WITH AUGMENTATION) ---
                if model_specific_transform and not config.get('augmentation', {}).get('enabled', False):
                    # Use timm's recommended transforms if augmentation is disabled
                    transform_train = model_specific_transform
                    transform_eval = model_specific_transform
                    print(f"   Using TIMM's recommended transforms (input size: {input_size}x{input_size}).")
                elif config.get('augmentation', {}).get('enabled', False):
                    # Use advanced augmentation strategy
                    print(f"   🎨 Using ADVANCED augmentation strategy: {config['augmentation'].get('strategy', 'light')}")
                    aug_strategy = create_augmentation_strategy(config, model_config)
                    transform_train = aug_strategy.get_train_transform()
                    transform_eval = aug_strategy.get_eval_transform()
                    print(f"   📈 Train augmentation: {len(transform_train.transforms)} steps")
                    print(f"   📊 Eval transforms: {len(transform_eval.transforms)} steps")
                else:
                    # Use default transforms
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
'''

# =============================================================================
# 3. MIXUP/CUTMIX SETUP
# =============================================================================

"""
Add this code after creating the dataloaders (around line 320):
"""

MIXUP_CUTMIX_SETUP = '''
                # --- Setup MixUp/CutMix (ADD THIS SECTION) ---
                mixup_cutmix = None
                aug_config = config.get('augmentation', {})
                if (aug_config.get('enabled', False) and 
                    (aug_config.get('mixup_alpha', 0) > 0 or aug_config.get('cutmix_alpha', 0) > 0)):
                    
                    print(f"\\n🎭 Setting up MixUp/CutMix augmentation...")
                    mixup_cutmix = MixupCutmixWrapper(
                        mixup_alpha=aug_config.get('mixup_alpha', 0.2),
                        cutmix_alpha=aug_config.get('cutmix_alpha', 1.0),
                        prob=aug_config.get('mixup_cutmix_prob', 1.0),
                        switch_prob=aug_config.get('switch_prob', 0.5),
                        num_classes=len(final_class_names),
                        label_smoothing=aug_config.get('label_smoothing', 0.1)
                    )
                    
                    if mixup_cutmix.is_enabled():
                        print(f"   ✅ MixUp/CutMix enabled")
                        print(f"   🔄 MixUp alpha: {aug_config.get('mixup_alpha', 0.2)}")
                        print(f"   ✂️ CutMix alpha: {aug_config.get('cutmix_alpha', 1.0)}")
                        print(f"   📊 Label smoothing: {aug_config.get('label_smoothing', 0.1)}")
                    else:
                        print(f"   ⚠️ MixUp/CutMix could not be enabled (timm required)")
                        mixup_cutmix = None
'''

# =============================================================================
# 4. TRAINING FUNCTION CALL UPDATE
# =============================================================================

"""
Update the train_model function call to include mixup_cutmix parameter:
"""

TRAINING_CALL_UPDATE = '''
                # Find the train_model call (around line 450) and add mixup_cutmix parameter:
                
                # OLD:
                # best_model, train_history = train_model(
                #     model, dataloaders, criterion, optimizer, scheduler, device,
                #     num_epochs=num_epochs, patience=patience, use_amp=use_amp,
                #     save_path=save_path, log_path=log_path, clip_grad_norm=clip_grad_norm,
                #     train_ratio=train_ratio, criterion_b=criterion_b, first_stage_epochs=first_stage_epochs
                # )
                
                # NEW:
                best_model, train_history = train_model(
                    model, dataloaders, criterion, optimizer, scheduler, device,
                    num_epochs=num_epochs, patience=patience, use_amp=use_amp,
                    save_path=save_path, log_path=log_path, clip_grad_norm=clip_grad_norm,
                    train_ratio=train_ratio, criterion_b=criterion_b, first_stage_epochs=first_stage_epochs,
                    mixup_cutmix=mixup_cutmix  # ADD THIS LINE
                )
'''

# =============================================================================
# 5. TRAINING.PY MODIFICATIONS
# =============================================================================

"""
Add this to the train_model function in src/training.py:
"""

TRAINING_PY_MODIFICATIONS = '''
# 1. Update function signature (around line 51):

def train_model(model, dataloaders, criterion, optimizer, scheduler, device,
                num_epochs=25, patience=5, use_amp=True, save_path='best_model.pth',
                log_path='training_log.csv', clip_grad_norm=1.0, train_ratio=1.0,
                criterion_b=None, first_stage_epochs=0, mixup_cutmix=None):  # ADD mixup_cutmix


# 2. In the training loop, after loading the batch (around line 175):

            for inputs, labels in pbar:
                # Skip batch if data loading failed
                if inputs.numel() == 0 or labels.numel() == 0:
                    warnings.warn(f"Skipping empty batch in {phase} phase (epoch {epoch+1}). Check data loading.")
                    continue

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Apply MixUp/CutMix if enabled and in training phase (ADD THIS)
                if phase == 'train' and mixup_cutmix and mixup_cutmix.is_enabled():
                    inputs, labels = mixup_cutmix(inputs, labels)

                # Zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)
                
                # ... rest of the training loop remains the same
'''

# =============================================================================
# 6. COMPLETE EXAMPLE CONFIG
# =============================================================================

EXAMPLE_CONFIG_YAML = '''
# Example augmentation configuration for config.yaml:

augmentation:
  enabled: true                     # Enable advanced augmentation
  strategy: 'medium'                # Augmentation intensity
  auto_augment_policy: 'original'   # AutoAugment policy
  randaugment_magnitude: 9          # RandAugment strength
  randaugment_num_ops: 2            # RandAugment operations per image
  mixup_alpha: 0.2                  # MixUp strength (0 = disabled)
  cutmix_alpha: 1.0                 # CutMix strength (0 = disabled)
  mixup_cutmix_prob: 1.0           # Probability of applying MixUp/CutMix
  switch_prob: 0.5                  # Mix between MixUp and CutMix
  random_erase_prob: 0.25           # Random erasing probability
  label_smoothing: 0.1              # Label smoothing for better generalization
  use_timm_auto_augment: true       # Use timm's implementation when available
  color_jitter: 0.4                 # Color augmentation strength
  scale_range: [0.7, 1.0]           # Random crop scale range
  ratio_range: [0.8, 1.2]           # Random crop aspect ratio range
'''

def print_integration_guide():
    """Print the complete integration guide."""
    
    print("🔧 ADVANCED AUGMENTATION INTEGRATION GUIDE")
    print("=" * 60)
    
    print("\n1️⃣  ADD IMPORTS TO main.py")
    print("-" * 30)
    print(IMPORTS_TO_ADD)
    
    print("\n2️⃣  REPLACE TRANSFORM CREATION")
    print("-" * 30)
    print("Replace the transform creation section in main.py with:")
    print(TRANSFORM_CREATION_CODE)
    
    print("\n3️⃣  ADD MIXUP/CUTMIX SETUP")
    print("-" * 30)
    print("Add after dataloader creation:")
    print(MIXUP_CUTMIX_SETUP)
    
    print("\n4️⃣  UPDATE TRAINING FUNCTION CALL")
    print("-" * 30)
    print(TRAINING_CALL_UPDATE)
    
    print("\n5️⃣  MODIFY TRAINING.PY")
    print("-" * 30)
    print(TRAINING_PY_MODIFICATIONS)
    
    print("\n6️⃣  UPDATE CONFIG.YAML")
    print("-" * 30)
    print("Add this augmentation section to your config.yaml:")
    print(EXAMPLE_CONFIG_YAML)
    
    print("\n✅ QUICK START GUIDE")
    print("=" * 60)
    print("1. Copy the imports to main.py")
    print("2. Replace transform creation logic")
    print("3. Add MixUp/CutMix setup")
    print("4. Update training function call")
    print("5. Modify training.py as shown")
    print("6. Update config.yaml with augmentation settings")
    print("7. Run training with: python main.py")
    
    print("\n🎯 STRATEGY RECOMMENDATIONS")
    print("=" * 60)
    print("• 'minimal': For small datasets or when overfitting")
    print("• 'light': Good starting point for most cases")
    print("• 'medium': For larger datasets, good balance")
    print("• 'heavy': When you need strong regularization")
    print("• 'extreme': For very large datasets or challenging problems")
    
    print("\n⚠️  IMPORTANT NOTES")
    print("=" * 60)
    print("• Install timm for best results: pip install timm")
    print("• Start with 'light' strategy and adjust based on results")
    print("• MixUp/CutMix work best with larger batch sizes (≥32)")
    print("• Monitor validation performance to avoid over-augmentation")
    print("• Some strategies may slow down training")


if __name__ == "__main__":
    print_integration_guide()
