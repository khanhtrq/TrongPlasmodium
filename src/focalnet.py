# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks
# This file now uses timm library to load FocalNet models
# The original implementation is available at: https://github.com/microsoft/FocalNet
# --------------------------------------------------------

import torch
import timm

# List of available FocalNet model variants in timm
AVAILABLE_MODELS = [
    "focalnet_tiny_srf",
    "focalnet_tiny_lrf",
    "focalnet_small_srf",
    "focalnet_small_lrf",
    "focalnet_base_srf",
    "focalnet_base_lrf",
    "focalnet_large_fl3",
    "focalnet_large_fl4",
    "focalnet_xlarge_fl3",
    "focalnet_xlarge_fl4",
    "focalnet_huge_fl3",
    "focalnet_huge_fl4"
]

def load_focalnet(model_name, pretrained=False, num_classes=1000):
    """
    Load a FocalNet model from timm
    
    Args:
        model_name (str): FocalNet variant name
        pretrained (bool): Whether to load pretrained weights
        num_classes (int): Number of output classes
    """
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        return model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print(f"Available FocalNet models in timm: {AVAILABLE_MODELS}")
        return None

# Convenience functions to maintain compatibility with existing code
def focalnet_tiny_srf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_tiny_srf", pretrained, **kwargs)

def focalnet_tiny_lrf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_tiny_lrf", pretrained, **kwargs)

def focalnet_small_srf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_small_srf", pretrained, **kwargs)

def focalnet_small_lrf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_small_lrf", pretrained, **kwargs)

def focalnet_base_srf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_base_srf", pretrained, **kwargs)

def focalnet_base_lrf(pretrained=False, **kwargs):
    return load_focalnet("focalnet_base_lrf", pretrained, **kwargs)

def focalnet_large_fl3(pretrained=False, **kwargs):
    return load_focalnet("focalnet_large_fl3", pretrained, **kwargs)

def focalnet_large_fl4(pretrained=False, **kwargs):
    return load_focalnet("focalnet_large_fl4", pretrained, **kwargs)

def focalnet_xlarge_fl3(pretrained=False, **kwargs):
    return load_focalnet("focalnet_xlarge_fl3", pretrained, **kwargs)

def focalnet_xlarge_fl4(pretrained=False, **kwargs):
    return load_focalnet("focalnet_xlarge_fl4", pretrained, **kwargs)

def focalnet_huge_fl3(pretrained=False, **kwargs):
    return load_focalnet("focalnet_huge_fl3", pretrained, **kwargs)

def focalnet_huge_fl4(pretrained=False, **kwargs):
    return load_focalnet("focalnet_huge_fl4", pretrained, **kwargs)

if __name__ == '__main__':
    # Test loading a model
    model_name = "focalnet_base_lrf"
    print(f"Testing model: {model_name}")
    
    try:
        # Check if timm is properly installed
        import timm
        print(f"timm version: {timm.__version__}")
        
        # Try loading without pretrained weights first (faster for testing)
        model = load_focalnet(model_name, pretrained=False, num_classes=1000)
        
        if model is not None:
            print(f"✓ Successfully loaded {model_name} from timm")
            
            # Create a test input
            img_size = 224
            x = torch.rand(1, 3, img_size, img_size)
            
            # Test forward pass
            with torch.no_grad():
                output = model(x)
            
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
            
            # Count parameters
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of parameters: {n_parameters:,}")
            
            # Test with pretrained weights
            print("\nTrying to load pretrained weights...")
            model = load_focalnet(model_name, pretrained=True, num_classes=5)
            # Test forward pass with pretrained weights
            with torch.no_grad():
                output = model(x)
            print(output)
            if model is not None:
                print(f"✓ Successfully loaded pretrained weights for {model_name}")
        else:
            print(f"✗ Failed to load {model_name}")
            
    except ImportError:
        print("✗ timm is not installed. Please install it with:")
        print("pip install timm")
    except Exception as e:
        print(f"✗ Error: {e}")
