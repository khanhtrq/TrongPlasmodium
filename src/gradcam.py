import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

from collections import defaultdict

def get_one_image_per_class(dataset):
    class_to_image = {}
    for img, label in dataset:
        if label not in class_to_image:
            class_to_image[label] = img.unsqueeze(0)
        if len(class_to_image) == len(dataset.classes):
            break
    return class_to_image

def setup_gradcam(model_name, model):
    model.eval().cuda()

    gradcam_data = {'features': None, 'gradients': None}

    def forward_hook(module, input, output):
        gradcam_data['features'] = output

    def backward_hook(module, grad_input, grad_output):
        gradcam_data['gradients'] = grad_output[0]

    if model_name == "resnet":
        target_layer = model.layer4[-1].conv3
    elif model_name == "alexnet":
        target_layer = model.features[-1]
    elif model_name == "vgg":
        target_layer = model.features[-1]
    elif model_name == "squeezenet":
        target_layer = model.features[-1]
    elif model_name == "densenet":
        target_layer = model.features[-1]
    elif model_name == "inception":
        target_layer = model.Mixed_7c
    elif model_name == "mobilenet_v2":
        target_layer = model.features[-1]
    elif model_name == "mobilenet_v3":
        target_layer = model.features[-1]
    else:
        raise ValueError(f"❌ Unsupported model: {model_name}")

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    print(f"[✅] Grad-CAM hook registered on {model_name.upper()} layer: {target_layer.__class__.__name__}")

    def compute_gradcam(input_tensor, class_idx=None):
        input_tensor = input_tensor.cuda()
        output = model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax().item()

        model.zero_grad()
        output[0, class_idx].backward()

        gradients = gradcam_data['gradients']
        features = gradcam_data['features']

        weights = torch.mean(gradients, dim=(2, 3))[0]
        cam = torch.zeros(features.shape[2:], dtype=torch.float32).cuda()

        for i, w in enumerate(weights):
            cam += w * features[0, i]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        return cam

    return model, gradcam_data, compute_gradcam

def show_gradcam_on_image(image_tensor, cam, title="Grad-CAM", save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    image_uint8 = np.uint8(image * 255)

    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image_uint8, 0.75, heatmap_color, 0.25, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap_color[..., ::-1])
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay[..., ::-1])
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[💾] Grad-CAM figure saved to: {save_path}")
        plt.close()
    else:
        plt.show()

import os
import torch.nn.functional as F

def generate_and_save_gradcam_per_class(model_name, model, dataset, transform, save_dir="gradcam_results"):
    os.makedirs(save_dir, exist_ok=True)

    class_to_image = get_one_image_per_class(dataset)

    model, gradcam_data, compute_gradcam = setup_gradcam(model_name, model)

    for class_idx, img in class_to_image.items():
        input_tensor = img.cuda()

        cam = compute_gradcam(input_tensor, class_idx=class_idx)

        class_name = dataset.classes[class_idx]
        save_path = os.path.join(save_dir, f"class_{class_idx}_{class_name}.jpg")

        show_gradcam_on_image(
            input_tensor,
            cam,
            title=f"Class {class_idx} - {class_name}",
            save_path=save_path
        )
