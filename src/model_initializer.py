import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import timm  # Add timm import


try:
    # Keep this for compatibility if focalnet is installed separately
    import focalnet as focalnet 
except ImportError:
    # Fallback to local focalnet if not installed globally
    try:
        import src.focalnet as focalnet
    except ImportError:
        print("⚠️ Warning: focalnet module not found locally or globally.")
        focalnet = None # Set to None if not found


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        print("🔒 Freezing base model parameters for feature extraction.")
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    """
    Initializes a model, prioritizing timm for loading.

    Args:
        model_name (str): Name of the model (e.g., 'resnet50', 'efficientnet_b0', 'focalnet_base_lrf').
        num_classes (int): Number of output classes.
        feature_extract (bool): If True, freeze base model weights.
        use_pretrained (bool): If True, load pretrained weights.

    Returns:
        tuple: (model, input_size, transform, model_config)
               - model: The initialized PyTorch model.
               - input_size: Recommended input image size (integer).
               - transform: Preprocessing transform pipeline from timm (or None).
               - model_config: Dictionary with model configuration details.
    """
    model_ft = None
    input_size = 224 # Default input size
    transform = None
    model_config = {}

    # --- Crucial Check: Ensure num_classes is valid ---
    if num_classes is None or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError(f"❌ num_classes must be a positive integer, but got {num_classes}")
    print(f"Initializing model '{model_name}' for {num_classes} classes...")

    # --- Try loading with timm first ---
    try:
        print(f"⏳ Attempting to load model '{model_name}' using timm...")
        model_ft = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=num_classes # Directly set the number of classes
        )
        print(f"✅ Successfully loaded '{model_name}' with timm.")

        # Apply feature extraction freezing if requested
        set_parameter_requires_grad(model_ft, feature_extract)

        # Get model-specific data configuration from timm
        data_config = timm.data.resolve_model_data_config(model_ft)
        transform = timm.data.create_transform(**data_config, is_training=False) # Get inference transform
        input_size = data_config.get('input_size', (3, 224, 224))[-1] # Get H or W

        model_config = {
            'input_size': input_size,
            'interpolation': data_config.get('interpolation', 'bicubic'),
            'mean': data_config.get('mean', (0.485, 0.456, 0.406)),
            'std': data_config.get('std', (0.229, 0.224, 0.225)),
            'crop_pct': data_config.get('crop_pct', 0.875) # Important for validation resizing
        }
        print(f"⚙️ timm model config: {model_config}")
        print(f"🖼️ Recommended input size: {input_size}x{input_size}")
        print(f"🔄 Using timm's recommended transform pipeline.")

    except Exception as e:
        print(f"⚠️ Failed to load '{model_name}' with timm: {e}. Falling back to torchvision or custom definitions...")

        # --- Fallback to torchvision models ---
        if model_name == "resnet":
            model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
        elif model_name == "alexnet":
            model_ft = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if use_pretrained else None)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224
        elif model_name == "vgg":
            model_ft = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1 if use_pretrained else None)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224
        elif model_name == "squeezenet":
            model_ft = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1 if use_pretrained else None)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224
        elif model_name == "densenet":
            model_ft = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if use_pretrained else None)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224
        elif model_name == "inception":
            model_ft = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1 if use_pretrained else None, aux_logits=True)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle main and aux classifiers
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299 # Specific to Inception v3
        elif model_name == "mobilenet_v2":
             model_ft = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if use_pretrained else None)
             set_parameter_requires_grad(model_ft, feature_extract)
             num_ftrs = model_ft.classifier[1].in_features
             model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
             input_size = 224
        elif model_name == "mobilenet_v3":
             model_ft = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if use_pretrained else None)
             set_parameter_requires_grad(model_ft, feature_extract)
             num_ftrs = model_ft.classifier[3].in_features
             # Replace the last layer (dropout + linear)
             model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
             input_size = 224
        else:
            model_ft = None # Model not found in timm or defined fallbacks

        if model_ft is None:
            print(f"❌ Error: Model '{model_name}' could not be initialized.")
            print("   Please check the model name and ensure 'timm' is installed (`pip install timm`).")
            print("   Available timm models can be listed with `timm.list_models('*keyword*')`.")
            exit() # Exit if model loading fails completely
        else:
            print(f"✅ Successfully loaded '{model_name}' using fallback definition.")
            print(f"🖼️ Using default input size: {input_size}x{input_size}")
            print(f"🔄 Using default transform pipeline (defined in main script).")
            # Populate model_config with defaults if not using timm
            model_config = {
                'input_size': input_size,
                'interpolation': 'bilinear', # Common default
                'mean': (0.485, 0.456, 0.406), # ImageNet defaults
                'std': (0.229, 0.224, 0.225),
                'crop_pct': 0.875 # Common default
            }

    return model_ft, input_size, transform, model_config

if __name__ == "__main__":

    # --- Test timm model ---
    model_name_timm = "mobilenetv4_hybrid_medium.e260_r256_in1k" # Example timm model
    num_classes_test = 5
    print(f"\n--- Testing timm model: {model_name_timm} ---")
    try:
        model_timm, input_size_timm, transform_timm, config_timm = initialize_model(
            model_name_timm, num_classes_test, feature_extract=False, use_pretrained=True
        )
        print(f"Loaded timm model: {model_name_timm}")
        print(f"Input Size: {input_size_timm}")
        print(f"Transform: {transform_timm}")
        print(f"Config: {config_timm}")

        # Test transform
        if transform_timm:
            from PIL import Image
            import numpy as np
            import matplotlib.pyplot as plt
            from urllib.request import urlopen

            print("\n=== Testing timm transform on image ===")
            try:
                img = Image.open(urlopen(
                    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
                )).convert('RGB')

                transformed_img = transform_timm(img)
                print(f"Transformed shape: {transformed_img.shape}")

            except Exception as img_err:
                print(f"Could not load or process test image: {img_err}")

    except Exception as e:
        print(f"Error testing timm model {model_name_timm}: {e}")

    # --- Test torchvision fallback ---
    model_name_tv = "resnet" # Use the keyword defined in the function
    print(f"\n--- Testing torchvision fallback: {model_name_tv} ---")
    try:
        model_tv, input_size_tv, transform_tv, config_tv = initialize_model(
            model_name_tv, num_classes_test, feature_extract=False, use_pretrained=True
        )
        print(f"Loaded torchvision model: {model_name_tv}")
        print(f"Input Size: {input_size_tv}")
        print(f"Transform: {transform_tv}") # Will be None
        print(f"Config: {config_tv}")
    except Exception as e:
        print(f"Error testing torchvision model {model_name_tv}: {e}")

    # --- Test invalid model name ---
    model_name_invalid = "non_existent_model_123"
    print(f"\n--- Testing invalid model name: {model_name_invalid} ---")
    try:
        # This should raise an error or exit
        initialize_model(model_name_invalid, num_classes_test)
    except SystemExit:
        print("Caught SystemExit as expected for invalid model.")
    except Exception as e:
        print(f"Caught unexpected error for invalid model: {e}")




