# Advanced Image Augmentation Module

Một module augmentation tiên tiến sử dụng thư viện timm để cải thiện hiệu suất training cho dự án PlasmodiumClassification.

## 🚀 Tính năng chính

### 1. Các chiến lược Augmentation
- **Minimal**: Augmentation cơ bản cho dataset nhỏ
- **Light**: Tốt cho hầu hết các trường hợp
- **Medium**: Cân bằng tốt cho dataset lớn hơn
- **Heavy**: Regularization mạnh
- **Extreme**: Cho dataset rất lớn hoặc bài toán khó

### 2. Kỹ thuật tiên tiến
- **AutoAugment**: Tự động tìm policy augmentation tối ưu
- **RandAugment**: Augmentation ngẫu nhiên với magnitude điều chỉnh được
- **MixUp**: Trộn images và labels
- **CutMix**: Cắt và dán các vùng của images
- **Random Erasing**: Xóa ngẫu nhiên các vùng trong image
- **Label Smoothing**: Làm mềm labels để tăng generalization

### 3. Tích hợp timm
- Sử dụng các transform được khuyến nghị bởi timm
- Tương thích với tất cả models timm
- Fallback về PyTorch transforms khi timm không có sẵn

## 📁 Cấu trúc files

```
src/
├── augment.py              # Module chính
examples/
├── augmentation_examples.py # Ví dụ sử dụng
├── integration_guide.py    # Hướng dẫn tích hợp
```

## 🔧 Cài đặt

```bash
# Cài đặt timm (khuyến nghị)
pip install timm

# Hoặc sử dụng với PyTorch transforms cơ bản
```

## 🎯 Sử dụng cơ bản

### 1. Tạo augmentation strategy

```python
from src.augment import TimmAugmentationStrategy

# Tạo strategy
aug_strategy = TimmAugmentationStrategy(
    strategy='light',
    input_size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

# Lấy transforms
train_transform = aug_strategy.get_train_transform()
eval_transform = aug_strategy.get_eval_transform()
```

### 2. Sử dụng với config

```python
from src.augment import create_augmentation_strategy

# Config trong YAML
config = {
    'augmentation': {
        'enabled': True,
        'strategy': 'medium',
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0
    }
}

# Tạo strategy từ config
aug_strategy = create_augmentation_strategy(config, model_config)
```

### 3. MixUp/CutMix trong training

```python
from src.augment import MixupCutmixWrapper

# Setup MixUp/CutMix
mixup_cutmix = MixupCutmixWrapper(
    mixup_alpha=0.2,
    cutmix_alpha=1.0,
    num_classes=6,
    label_smoothing=0.1
)

# Trong training loop
for inputs, labels in dataloader:
    if mixup_cutmix.is_enabled():
        inputs, labels = mixup_cutmix(inputs, labels)
    
    # Training step bình thường...
```

## ⚙️ Cấu hình trong config.yaml

```yaml
augmentation:
  enabled: true                     # Bật/tắt augmentation
  strategy: 'medium'                # Mức độ augmentation
  auto_augment_policy: 'original'   # Policy AutoAugment
  randaugment_magnitude: 9          # Độ mạnh RandAugment
  randaugment_num_ops: 2            # Số operations RandAugment
  mixup_alpha: 0.2                  # Tham số MixUp
  cutmix_alpha: 1.0                 # Tham số CutMix
  mixup_cutmix_prob: 1.0           # Xác suất áp dụng MixUp/CutMix
  switch_prob: 0.5                  # Xác suất chuyển đổi MixUp/CutMix
  random_erase_prob: 0.25           # Xác suất Random Erasing
  label_smoothing: 0.1              # Label smoothing
  color_jitter: 0.4                 # Độ mạnh color jitter
  scale_range: [0.7, 1.0]           # Phạm vi scale cho random crop
  ratio_range: [0.8, 1.2]           # Phạm vi aspect ratio
```

## 🔄 Tích hợp vào main.py

### 1. Thêm imports

```python
from src.augment import (
    create_augmentation_strategy,
    MixupCutmixWrapper,
    get_timm_transform
)
```

### 2. Cập nhật tạo transforms

```python
# Thay thế logic tạo transform hiện tại
if config.get('augmentation', {}).get('enabled', False):
    aug_strategy = create_augmentation_strategy(config, model_config)
    transform_train = aug_strategy.get_train_transform()
    transform_eval = aug_strategy.get_eval_transform()
else:
    # Sử dụng transforms mặc định
    transform_train = default_train_transform
    transform_eval = default_eval_transform
```

### 3. Setup MixUp/CutMix

```python
# Sau khi tạo dataloaders
mixup_cutmix = None
aug_config = config.get('augmentation', {})
if aug_config.get('enabled', False) and aug_config.get('mixup_alpha', 0) > 0:
    mixup_cutmix = MixupCutmixWrapper(
        mixup_alpha=aug_config.get('mixup_alpha', 0.2),
        cutmix_alpha=aug_config.get('cutmix_alpha', 1.0),
        num_classes=len(final_class_names),
        label_smoothing=aug_config.get('label_smoothing', 0.1)
    )
```

### 4. Cập nhật training function

```python
# Thêm parameter mixup_cutmix vào train_model call
best_model, train_history = train_model(
    model, dataloaders, criterion, optimizer, scheduler, device,
    # ... các parameters khác
    mixup_cutmix=mixup_cutmix  # Thêm dòng này
)
```

### 5. Sửa training.py

```python
# Trong function signature
def train_model(..., mixup_cutmix=None):

# Trong training loop
for inputs, labels in pbar:
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Thêm dòng này
    if phase == 'train' and mixup_cutmix and mixup_cutmix.is_enabled():
        inputs, labels = mixup_cutmix(inputs, labels)
    
    # Tiếp tục training bình thường...
```

## 📊 Khuyến nghị strategies

| Strategy | Use Case | Đặc điểm |
|----------|----------|----------|
| `minimal` | Dataset nhỏ, overfitting | Chỉ flip + resize |
| `light` | Hầu hết trường hợp | Geometric + color cơ bản |
| `medium` | Dataset trung bình | Augmentation cân bằng |
| `heavy` | Dataset lớn, cần regularization | AutoAugment + Random Erasing |
| `extreme` | Dataset rất lớn | Tất cả techniques |

## 🧪 Testing

```bash
# Test module
python src/augment.py

# Chạy examples
python examples/augmentation_examples.py

# Xem hướng dẫn tích hợp
python examples/integration_guide.py
```

## ⚠️ Lưu ý quan trọng

1. **Cài đặt timm**: `pip install timm` để có hiệu suất tốt nhất
2. **Batch size**: MixUp/CutMix hoạt động tốt với batch size ≥ 32
3. **Monitor performance**: Theo dõi validation để tránh over-augmentation
4. **Training time**: Một số strategies có thể làm chậm training
5. **Memory**: Heavy/extreme strategies cần nhiều memory hơn

## 🔬 Các kỹ thuật chi tiết

### AutoAugment
Tự động tìm kiếm policy augmentation tối ưu cho dataset cụ thể.

### RandAugment  
Chọn ngẫu nhiên N operations từ một pool với magnitude M.

### MixUp
Trộn tuyến tính hai images và labels: `new_image = λ * img1 + (1-λ) * img2`

### CutMix
Cắt và dán vùng rectangular giữa hai images, labels tỷ lệ theo diện tích.

### Random Erasing
Xóa ngẫu nhiên vùng rectangular trong image để tăng robustness.

## 🎨 Examples

Xem thêm trong `examples/` folder:
- `augmentation_examples.py`: Ví dụ sử dụng từng tính năng
- `integration_guide.py`: Hướng dẫn tích hợp chi tiết

## 🤝 Contributing

1. Test kỹ trước khi commit
2. Thêm examples cho features mới
3. Update documentation
4. Maintain backward compatibility

---

Được tạo bởi GitHub Copilot - May 2025
