import torch
import torch.nn as nn

def get_device(use_cuda=True, multi_gpu=True):
    """
    Configure and return the device to use for model training.
    
    Args:
        use_cuda (bool): Whether to use CUDA if available
        multi_gpu (bool): Whether to use multiple GPUs if available
    
    Returns:
        device: PyTorch device to use
        gpu_count: Number of GPUs being used
    """
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available() and use_cuda
    
    # Get device
    device = torch.device("cuda" if cuda_available else "cpu")
    
    # Get GPU count
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    
    return device, gpu_count

def setup_model_for_training(model, use_cuda=True, multi_gpu=True):
    """
    Set up a model for training on the appropriate device(s).
    
    Args:
        model: PyTorch model
        use_cuda (bool): Whether to use CUDA if available
        multi_gpu (bool): Whether to use multiple GPUs if available
    
    Returns:
        model: Model prepared for training (possibly wrapped in DataParallel)
        device: Device the model is on
    """
    device, gpu_count = get_device(use_cuda, multi_gpu)
    
    # Move model to device
    model = model.to(device)
    
    # Use DataParallel if multiple GPUs available and multi_gpu is enabled
    if cuda_available := (torch.cuda.is_available() and use_cuda):
        if gpu_count > 1 and multi_gpu:
            print(f"⚡ Enabling DataParallel across {gpu_count} GPUs")
            model = nn.DataParallel(model)
    
    return model, device
