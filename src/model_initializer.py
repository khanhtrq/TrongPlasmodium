import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import timm  # Add timm import


try:
    import focalnet as focalnet  # Keep this for compatibility with existing code
except ImportError:
    import src.focalnet as focalnet  # Keep this for compatibility with existing code
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    model_ft = None
    input_size = 0
    transform = None
    model_config = {}

    # --- DEBUG: Ensure num_classes is valid ---
    if num_classes is None or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes}")

    if model_name == "resnet":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name == "mobilenet_v2":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mobilenet_v3":
        model_ft = models.mobilenet_v3_large(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        # Try loading any model with timm
        try:
            print(f"Trying to load model '{model_name}' with timm...")
            model_ft = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=num_classes
            )
            set_parameter_requires_grad(model_ft, feature_extract)

            data_config = timm.data.resolve_model_data_config(model_ft)
            transform = timm.data.create_transform(**data_config, is_training=False)
            input_size = data_config.get('input_size', (3, 224, 224))[-1]

            model_config = {
            'input_size': input_size,
            'crop_pct': data_config.get('crop_pct', 0.875),
            'mean': data_config.get('mean', (0.485, 0.456, 0.406)),
            'std': data_config.get('std', (0.229, 0.224, 0.225)),
            'interpolation': data_config.get('interpolation', 'bicubic')
            }
            print(f"Model config: {model_config}")

        except Exception as e:
            print(f"❌ Error loading model '{model_name}' with timm: {e}")
            print("Make sure timm is installed: pip install timm")
            print("Available models in timm with this prefix:")
            matching_models = [m for m in timm.list_models() if model_name.split('_')[0] in m][:10]
            for m in matching_models:
                print(f"  - {m}")
            exit()

    print(model_ft)
    return model_ft, input_size, transform, model_config

if __name__ == "__main__":
    
    model_name = "ese_vovnet57b"  # Example model name
    num_classes = 5  # Số lớp ví dụ
    feature_extract = False
    use_pretrained = True

    model, input_size, transform, model_config = initialize_model(model_name, num_classes, feature_extract, use_pretrained)
    print(f"Loaded model: {model_name} with input size {input_size}")
    print(f"Transform pipeline: {transform}")
    print(f"Model config: {model_config}")
    
    if transform:
        from PIL import Image
        import numpy as np
        import matplotlib.pyplot as plt
        from urllib.request import urlopen
        
        print("\n=== Testing transform on image ===")
        img = Image.open(urlopen(
            'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
        ))
        
        transformed_img = transform(img)
        print(f"Transformed shape: {transformed_img.shape}")

        # Show original and transformed image side by side
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # Convert transformed tensor to numpy image for display
        npimg = transformed_img.numpy()
        if npimg.shape[0] == 1:  # grayscale
            npimg = npimg.squeeze(0)
        else:
            npimg = np.transpose(npimg, (1, 2, 0))
        # Undo normalization for display if mean/std in model_config
        mean = model_config.get('mean', (0.485, 0.456, 0.406))
        std = model_config.get('std', (0.229, 0.224, 0.225))
        npimg = npimg * std + mean
        npimg = np.clip(npimg, 0, 1)
        axs[1].imshow(npimg)
        axs[1].set_title("Transformed Image")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()




