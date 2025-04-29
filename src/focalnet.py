# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks
# This file now primarily acts as a wrapper to load FocalNet models using the timm library.
# Original implementation: https://github.com/microsoft/FocalNet
# timm library: https://github.com/huggingface/pytorch-image-models
# --------------------------------------------------------

import torch
import timm
import warnings

# List of known FocalNet model variants available in timm (as of timm ~0.9.x)
# You can get an updated list using: `timm.list_models('focalnet*')`
AVAILABLE_FOCALNET_MODELS = [
    "focalnet_tiny_srf",    # FocalNet-T (SRF) ImageNet-1k
    "focalnet_tiny_lrf",    # FocalNet-T (LRF) ImageNet-1k
    "focalnet_small_srf",   # FocalNet-S (SRF) ImageNet-1k
    "focalnet_small_lrf",   # FocalNet-S (LRF) ImageNet-1k
    "focalnet_base_srf",    # FocalNet-B (SRF) ImageNet-1k
    "focalnet_base_lrf",    # FocalNet-B (LRF) ImageNet-1k
    "focalnet_large_fl3",   # FocalNet-L (FL=3) ImageNet-22k -> 1k
    "focalnet_large_fl4",   # FocalNet-L (FL=4) ImageNet-22k -> 1k
    "focalnet_xlarge_fl3",  # FocalNet-XL (FL=3) ImageNet-22k -> 1k
    "focalnet_xlarge_fl4",  # FocalNet-XL (FL=4) ImageNet-22k -> 1k
    "focalnet_huge_fl3",    # FocalNet-H (FL=3) ImageNet-22k -> 1k
    "focalnet_huge_fl4",    # FocalNet-H (FL=4) ImageNet-22k -> 1k
]

def load_focalnet(model_name, pretrained=False, num_classes=1000, **kwargs):
    """
    Loads a FocalNet model variant using the timm library.

    Args:
        model_name (str): The specific FocalNet variant name (e.g., 'focalnet_base_lrf').
                          Must be a valid timm model name.
        pretrained (bool): Whether to load pretrained weights (usually ImageNet).
                           Defaults to False.
        num_classes (int): Number of output classes for the final classifier.
                           Defaults to 1000 (ImageNet).
        **kwargs: Additional keyword arguments passed directly to `timm.create_model`.

    Returns:
        torch.nn.Module: The loaded FocalNet model.
        None: If the model name is invalid or loading fails.
    """
    if model_name not in AVAILABLE_FOCALNET_MODELS:
         warnings.warn(f"⚠️ Model name '{model_name}' might not be a standard timm FocalNet name. "
                       f"Known names: {AVAILABLE_FOCALNET_MODELS}. Attempting to load anyway.")
         # Check if it's available in the current timm version
         if not timm.is_model(model_name):
              print(f"❌ Error: Model '{model_name}' is not available in the current timm installation.")
              print(f"   Available models matching 'focalnet*': {timm.list_models('focalnet*')}")
              return None


    try:
        print(f"⏳ Loading FocalNet model '{model_name}' using timm (pretrained={pretrained}, num_classes={num_classes})...")
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs
        )
        print(f"✅ Successfully loaded '{model_name}'.")
        return model
    except Exception as e:
        print(f"❌ Error loading model '{model_name}' with timm: {e}")
        print(f"   Ensure 'timm' is installed and up-to-date (`pip install --upgrade timm`).")
        return None

# --- Convenience functions (Optional - maintain compatibility) ---
# These functions simply call the main loader.

def focalnet_tiny_srf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_tiny_srf", pretrained=pretrained, **kwargs)

def focalnet_tiny_lrf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_tiny_lrf", pretrained=pretrained, **kwargs)

def focalnet_small_srf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_small_srf", pretrained=pretrained, **kwargs)

def focalnet_small_lrf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_small_lrf", pretrained=pretrained, **kwargs)

def focalnet_base_srf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_base_srf", pretrained=pretrained, **kwargs)

def focalnet_base_lrf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_base_lrf", pretrained=pretrained, **kwargs)

def focalnet_large_fl3(pretrained=False, **kwargs):
    return load_focalnet("focalnet_large_fl3", pretrained=pretrained, **kwargs)

def focalnet_large_fl4(pretrained=False, **kwargs):
    return load_focalnet("focalnet_large_fl4", pretrained=pretrained, **kwargs)

def focalnet_xlarge_fl3(pretrained=False, **kwargs):
    return load_focalnet("focalnet_xlarge_fl3", pretrained=pretrained, **kwargs)

def focalnet_xlarge_fl4(pretrained=False, **kwargs):
    return load_focalnet("focalnet_xlarge_fl4", pretrained=pretrained, **kwargs)

def focalnet_huge_fl3(pretrained=False, **kwargs):
    return load_focalnet("focalnet_huge_fl3", pretrained=pretrained, **kwargs)

def focalnet_huge_fl4(pretrained=False, **kwargs):
    return load_focalnet("focalnet_huge_fl4", pretrained=pretrained, **kwargs)


# --- Example Usage ---
if __name__ == '__main__':
    print("-" * 30)
    print("Testing FocalNet loading via timm wrapper")
    print("-" * 30)

    # Test case 1: Load a specific model without pretrained weights, custom classes
    model_name_test = "focalnet_small_lrf"
    num_classes_test = 10
    print(f"\nAttempting to load: {model_name_test} (pretrained=False, num_classes={num_classes_test})")
    model1 = load_focalnet(model_name_test, pretrained=False, num_classes=num_classes_test)

    if model1:
        print(f"✓ Model loaded.")
        # Basic check: Input/Output shape
        try:
            # Get input size from model config (usually 224 for FocalNets)
            data_config = timm.data.resolve_model_data_config(model1)
            input_size = data_config.get('input_size', (3, 224, 224))
            img_size = input_size[-1]

            dummy_input = torch.randn(2, 3, img_size, img_size) # Batch size 2
            with torch.no_grad():
                output = model1(dummy_input)
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}") # Should be (2, num_classes_test)
            assert output.shape == (2, num_classes_test)
            print("   Input/Output shape test PASSED.")
        except Exception as e:
            print(f"   Input/Output shape test FAILED: {e}")

        # Count parameters
        n_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
        print(f"   Number of trainable parameters: {n_params:,}")
    else:
        print(f"✗ Failed to load {model_name_test}.")

    print("-" * 30)

    # Test case 2: Load with pretrained weights (ImageNet default classes)
    model_name_pretrained = "focalnet_base_lrf"
    print(f"\nAttempting to load: {model_name_pretrained} (pretrained=True, num_classes=1000)")
    # Use the convenience function for this one
    model2 = focalnet_base_lrf(pretrained=True) # num_classes defaults to 1000

    if model2:
        print(f"✓ Pretrained model loaded.")
        n_params_pre = sum(p.numel() for p in model2.parameters())
        print(f"   Total parameters: {n_params_pre:,}")
    else:
        print(f"✗ Failed to load pretrained {model_name_pretrained}.")

    print("-" * 30)

    # Test case 3: Invalid model name
    invalid_name = "focalnet_nonexistent_model"
    print(f"\nAttempting to load invalid name: {invalid_name}")
    model3 = load_focalnet(invalid_name)
    if model3 is None:
        print("✓ Correctly failed to load invalid model name.")
    else:
        print("✗ Incorrectly loaded an invalid model name (or timm behavior changed).")

    print("-" * 30)
