import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import timm  # Add timm import

import src.focalnet as focalnet  # Keep this for compatibility with existing code

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    model_ft = None
    input_size = 0

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

    elif model_name.startswith("focalnet"):
        try:
            # Use timm directly to load the model
            model_ft = timm.create_model(
                model_name,  # Model name like "focalnet_base_lrf" 
                pretrained=use_pretrained,
                num_classes=num_classes
            )
            # Feature extraction mode if requested
            set_parameter_requires_grad(model_ft, feature_extract)
            
            # Standard input size for FocalNet models
            input_size = 224
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            print("Make sure timm is installed: pip install timm")
            print("Available FocalNet models in timm:")
            focalnet_models = [m for m in timm.list_models() if "focalnet" in m]
            for m in focalnet_models:
                print(f"  - {m}")
            exit()

    else:
        print("Invalid model name, exiting...")
        exit()

    print(model_ft)
    return model_ft, input_size

if __name__ == "__main__":
    # Test load focalnet_base_lrf
    model_name = "focalnet_base_lrf"
    num_classes = 5  # Số lớp ví dụ
    feature_extract = False
    use_pretrained = True

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained)
    print(f"Loaded model: {model_name} with input size {input_size}")

