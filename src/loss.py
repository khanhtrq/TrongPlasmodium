import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        if self.alpha is not None:
            try:
                at = self.alpha.gather(0, targets)
            except:
                at = 1.0
        else:
            at = 1.0
    
        loss = -at * (1 - pt) ** self.gamma * logpt
    
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def compute_alpha_from_dataloader(train_loader, num_classes, device):
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=np.array(all_labels)
    )
    
    alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)
    return alpha

def compute_class_weights(dataloader, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in dataloader:
        for label in labels:
            counts[label] += 1
    total = counts.sum()
    weights = total / (counts + 1e-6)
    return weights.to(device)


class F1Loss(nn.Module):
    '''
    Computes the F1 loss, a differentiable approximation of the F1 score.
    Assumes inputs are raw logits and targets are class indices.

    Note on Macro Averaging Behavior:
    The F1 score is calculated per class and then averaged (macro F1).
    If a class is not present in the `targets` of a given batch:
    - Its True Positives (TP) and False Negatives (FN) will be 0.
    - If the model correctly assigns low probability to this absent class, its False Positives (FP) will also be near 0.
    - This results in Precision = 0 / (0 + near_0_FP + epsilon) approx 0, and Recall = 0 / (0 + 0 + epsilon) = 0.
    - Consequently, the F1 score for such an absent class will be 0.
    When these per-class F1 scores are averaged, the 0s from absent classes will pull down
    the overall macro F1 score. Therefore, even if all *present* classes in a batch
    are predicted perfectly (F1=1 for them), the total `f1.mean()` might be less than 1,
    and thus `F1Loss = 1 - f1.mean()` might be greater than 0.
    For example, with 4 classes, if 3 are present and perfectly predicted (F1=1 each)
    and 1 class is absent (contributing F1=0 to the macro average), the `f1.mean()`
    would be (1+1+1+0)/4 = 0.75, leading to an F1Loss of 0.25.
    This is standard behavior for macro-averaged F1 scores.
    '''
    def __init__(self, num_classes, epsilon=1e-7, beta=1.0, reduction='mean'):
        super(F1Loss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (batch_size, num_classes) - Raw logits from the model
        # targets: (batch_size) - Long tensor of class indices

        # Apply softmax to get probabilities
        probas = F.softmax(inputs, dim=1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).to(dtype=inputs.dtype)
        # targets_one_hot: (batch_size, num_classes)

        # Calculate true positives, false positives, false negatives per class
        # These are sums over the batch dimension
        tp = (probas * targets_one_hot).sum(dim=0)
        fp = (probas * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - probas) * targets_one_hot).sum(dim=0)

        # Calculate precision and recall per class
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        # Calculate F1 score per class
        f1 = (1 + self.beta**2) * (precision * recall) / ((self.beta**2 * precision) + recall + self.epsilon)

        # Average F1 score across classes (macro F1)
        # The loss is 1 - F1 score
        f1_loss = 1 - f1.mean() # Macro F1 loss

        # Note: Reduction parameter isn't explicitly used here as we default to macro average loss.
        # You could adapt this part if 'sum' or element-wise loss is needed.
        return f1_loss


class GHMCClassificationLoss(nn.Module):
    def __init__(self, bins=10, momentum=0.0, loss_weight=1.0):
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.loss_weight = loss_weight

        self.edges = torch.arange(bins + 1).float() / bins
        self.edges[-1] += 1e-6  # để tránh trùng ranh giới

        if momentum > 0:
            self.acc_sum = torch.zeros(bins)
        else:
            self.acc_sum = None

    def forward(self, logits, target_indices):
        """
        logits: Tensor [B, C] — raw output từ model (chưa sigmoid)
        target_indices: Tensor [B] — class index (giống CrossEntropyLoss)
        """
        device = logits.device
        B, C = logits.shape

        # Đưa các tensor phụ về đúng device
        edges = self.edges.to(device)
        if self.acc_sum is not None and self.acc_sum.device != device:
            self.acc_sum = self.acc_sum.to(device)

        # Chuyển target thành one-hot [B, C]
        target = F.one_hot(target_indices, num_classes=C).float()

        # Label weight mặc định là toàn 1
        label_weight = torch.ones_like(target)

        # Gradient length
        g = torch.abs(logits.sigmoid().detach() - target)

        weights = torch.zeros_like(logits)
        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0

        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1

        if n > 0:
            weights = weights / n

        # Loss tính theo BCE logits
        loss = F.binary_cross_entropy_with_logits(logits, target, weights, reduction='sum') / tot
        return loss * self.loss_weight
    
def get_criterion(criterion, num_classes, device, criterion_params=None):
    criterion = criterion.lower() if isinstance(criterion, str) else criterion
    criterion_params = criterion_params or {}
    print(f"Using criterion: {criterion} with params: {criterion_params}")
    if criterion == 'focalloss':
        return FocalLoss(
            alpha=criterion_params.get('alpha', 1.0),
            gamma=criterion_params.get('gamma', 2.0),
            reduction=criterion_params.get('reduction', 'mean')
        ).to(device)
    elif criterion == 'f1loss':
        return F1Loss(
            num_classes=num_classes,
            epsilon=criterion_params.get('epsilon', 1e-7),
            beta=criterion_params.get('beta', 1.0),
            reduction=criterion_params.get('reduction', 'mean')
        ).to(device)
    elif criterion in ['ghmc', 'ghmcclassificationloss']:
        return GHMCClassificationLoss(
            bins=criterion_params.get('bins', 10),
            momentum=criterion_params.get('momentum', 0.0),
            loss_weight=criterion_params.get('loss_weight', 1.0)
        ).to(device)
    else:
        ce_kwargs = {}
        if 'weight' in criterion_params:
            ce_kwargs['weight'] = criterion_params['weight']
        if 'reduction' in criterion_params:
            ce_kwargs['reduction'] = criterion_params['reduction']
        return nn.CrossEntropyLoss(**ce_kwargs).to(device)

if __name__ == "__main__":
    # Test các loss function với dữ liệu giả lập
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4
    batch_size = 8

    # Tạo logits và targets giả lập
    logits = torch.randn(batch_size, num_classes, requires_grad=True).to(device)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)

    print("Logits:\n", logits)
    print("Targets:\n", targets)

    # Test FocalLoss
    print("\nTesting FocalLoss:")
    focal_loss = get_criterion('focalloss', num_classes, device)
    loss_focal = focal_loss(logits, targets)
    print("FocalLoss:", loss_focal.item())

    # Test F1Loss
    print("\nTesting F1Loss:")
    f1_loss = get_criterion('f1loss', num_classes, device)
    loss_f1 = f1_loss(logits, targets)
    print("F1Loss:", loss_f1.item())

    # Test CrossEntropyLoss
    print("\nTesting CrossEntropyLoss:")
    ce_loss = get_criterion('crossentropy', num_classes, device)
    loss_ce = ce_loss(logits, targets)
    print("CrossEntropyLoss:", loss_ce.item())

    # Test GHMCClassificationLoss (dùng CPU nếu không có CUDA)
    print("\nTesting GHMCClassificationLoss:")
    ghmc_loss_fn = get_criterion('ghmc', num_classes, device)
    # Chuẩn bị target dạng index (giống CrossEntropy)
    ghmc_loss = ghmc_loss_fn(logits, targets)
    print("GHMCClassificationLoss:", ghmc_loss.item())

    # ==== Trường hợp lý tưởng: tất cả predict đều đúng ====
    print("\n=== Ideal case: All predictions correct ===")
    # Logits: mỗi sample, đúng class có giá trị lớn nhất (ví dụ: 10.0), các class khác giá trị nhỏ (ví dụ: -10.0)
    targets_perfect = torch.randint(0, num_classes, (batch_size,), device=device) # Giữ targets ngẫu nhiên
    logits_perfect = torch.full((batch_size, num_classes), -10.0, device=device, dtype=torch.float32)
    logits_perfect[torch.arange(batch_size), targets_perfect] = 10.0
    logits_perfect.requires_grad_(True)


    print("Targets (perfect):", targets_perfect)
    # print("Logits (perfect):\n", logits_perfect) # Có thể bỏ comment để xem

    print(f"FocalLoss (perfect): {focal_loss(logits_perfect, targets_perfect).item():.4f}")
    print(f"F1Loss (perfect): {f1_loss(logits_perfect, targets_perfect).item():.4f}")
    print(f"CrossEntropyLoss (perfect): {ce_loss(logits_perfect, targets_perfect).item():.4f}")
    print(f"GHMCClassificationLoss (perfect): {ghmc_loss_fn(logits_perfect, targets_perfect).item():.4f}")

    # ==== Trường hợp lý tưởng: tất cả predict đều sai ====
    print("\n=== Worst case: All predictions wrong ===")
    # Logits: mỗi sample, một class sai nào đó có giá trị lớn nhất
    targets_wrong = torch.randint(0, num_classes, (batch_size,), device=device) # Giữ targets ngẫu nhiên
    logits_wrong = torch.full((batch_size, num_classes), -10.0, device=device, dtype=torch.float32)
    # Chọn một class sai ngẫu nhiên cho mỗi sample để đặt giá trị cao
    wrong_class_indices = (targets_wrong + torch.randint(1, num_classes, (batch_size,), device=device)) % num_classes
    logits_wrong[torch.arange(batch_size), wrong_class_indices] = 10.0
    logits_wrong.requires_grad_(True)

    print("Targets (wrong):", targets_wrong)
    # print("Logits (wrong):\n", logits_wrong) # Có thể bỏ comment để xem

    print(f"FocalLoss (wrong): {focal_loss(logits_wrong, targets_wrong).item():.4f}")
    print(f"F1Loss (wrong): {f1_loss(logits_wrong, targets_wrong).item():.4f}")
    print(f"CrossEntropyLoss (wrong): {ce_loss(logits_wrong, targets_wrong).item():.4f}")
    print(f"GHMCClassificationLoss (wrong): {ghmc_loss_fn(logits_wrong, targets_wrong).item():.4f}")