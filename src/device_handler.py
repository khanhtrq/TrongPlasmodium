import torch

try:
    import torch_xla.core.xla_model as xm
    _tpu_available = True
except ImportError:
    _tpu_available = False

def get_device():
    if _tpu_available:
        return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
