import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
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
        self.epsilon = float(epsilon)  # Ensure epsilon is a float
        self.beta = float(beta)        # Ensure beta is a float
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
    def __init__(self, bins=10, momentum=0.1, loss_weight=1.0):
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

def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return loss.mean()

class IBLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000.):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)),1) # N * 1
        ib = grads*features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon)
        return ib_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib)

def ib_focal_loss(input_values, ib, gamma):
    """Computes the ib focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values * ib
    return loss.mean()

class IB_FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000., gamma=0.):
        super(IB_FocalLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)),1) # N * 1
        ib = grads*(features.reshape(-1))
        ib = self.alpha / (ib + self.epsilon)
        return ib_focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma)

class LDAMLoss(nn.Module):
    """
    LDAMLoss for classification, compatible with nn.CrossEntropyLoss input format.
    Args:
        cls_num_list: List or array of number of samples per class.
        max_m: Maximum margin.
        weight: Optional tensor of per-class weights.
        s: Scaling factor.
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)  # Store as CPU tensor, move to device in forward
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, input, target):
        """
        Args:
            input: [batch_size, num_classes] logits (same as nn.CrossEntropyLoss)
            target: [batch_size] integer class indices
        Returns:
            Scalar loss
        """
        if input.device != self.m_list.device:
            m_list = self.m_list.to(input.device)
        else:
            m_list = self.m_list

        # Create margin tensor for each sample in the batch
        batch_m = m_list[target]  # [batch_size]
        # Subtract margin only from the true class logits
        output = input.clone()
        output[torch.arange(input.size(0)), target] -= batch_m

        return F.cross_entropy(self.s * output, target, weight=self.weight)

# Helper function for ClassBalancedLoss
def _cb_focal_loss(labels_one_hot, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels_one_hot`."""    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels_one_hot, reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels_one_hot * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss
    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)
    focal_loss /= torch.sum(labels_one_hot)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`."""
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights_per_cls = (1.0 - beta) / np.array(effective_num)
    weights_per_cls = weights_per_cls / np.sum(weights_per_cls) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float().to(logits.device)
    weights_tensor = torch.tensor(weights_per_cls, dtype=torch.float32, device=logits.device)
    cb_weights = weights_tensor[labels].unsqueeze(1).repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = _cb_focal_loss(labels_one_hot, logits, cb_weights, gamma)
    elif loss_type == "sigmoid":
        bce_loss_terms = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=cb_weights, reduction="sum")
        cb_loss = bce_loss_terms / labels_one_hot.shape[0]
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        bce_on_softmax_terms = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=cb_weights, reduction="sum")
        cb_loss = bce_on_softmax_terms / labels_one_hot.shape[0]
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type} for CB_loss")
        
    return cb_loss

class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_cls, num_classes, loss_type='focal', beta=0.9999, gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        if not isinstance(samples_per_cls, (list, np.ndarray, torch.Tensor)):
            raise TypeError("samples_per_cls must be a list, numpy array or torch tensor.")
        if len(samples_per_cls) != num_classes:
            raise ValueError(f"Length of samples_per_cls ({len(samples_per_cls)}) must be equal to num_classes ({num_classes}).")

    def forward(self, logits, target):
        return CB_loss(target, logits, self.samples_per_cls, self.num_classes,
                       self.loss_type, self.beta, self.gamma)

from torch.distributions import MultivariateNormal as MVN

def bmc_loss_md(pred, target, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`."""
    I = torch.eye(pred.shape[-1], device=pred.device)
    logits = MVN(pred.unsqueeze(1), noise_var * I).log_prob(target.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=pred.device))
    loss = loss * (2 * noise_var).detach()
    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma=8.):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, input, target):
        print(input, target)
        num_classes = input.size(1)
        target_onehot = F.one_hot(target, num_classes=num_classes).float()
        noise_var = self.noise_sigma ** 2
        return bmc_loss_md(input, target_onehot, noise_var)

def get_active_criterion(epoch, criterion_a, criterion_b=None, first_stage_epochs=0):
    if criterion_b is not None and first_stage_epochs > 0 and epoch >= first_stage_epochs:
        return criterion_b
    return criterion_a
    
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
    elif criterion == 'ibloss':
        return IBLoss(
            weight=criterion_params.get('weight', None),
            alpha=float(criterion_params.get('alpha', 10000.))
        ).to(device)
    elif criterion == 'ib_focalloss':
        return IB_FocalLoss(
            weight=criterion_params.get('weight', None),
            alpha=float(criterion_params.get('alpha', 10000.)),
            gamma=float(criterion_params.get('gamma', 2.))
        ).to(device)
    elif criterion == 'bmcloss':
        return BMCLoss(
            init_noise_sigma=float(criterion_params.get('init_noise_sigma', 8.))
        ).to(device)
    elif criterion == 'ldamloss':
        return LDAMLoss(
            cls_num_list=criterion_params.get('cls_num_list', [1]*num_classes),
            max_m=float(criterion_params.get('max_m', 0.5)),
            weight=criterion_params.get('weight', None),
            s=float(criterion_params.get('s', 30))
        ).to(device)
    elif criterion in ['cbloss', 'classbalancedloss']:
        samples_per_cls = criterion_params.get('samples_per_cls')
        if samples_per_cls is None:
            raise ValueError("samples_per_cls must be provided for ClassBalancedLoss")
        return ClassBalancedLoss(
            samples_per_cls=samples_per_cls,
            num_classes=num_classes,
            loss_type=criterion_params.get('loss_type', 'focal'),
            beta=float(criterion_params.get('beta', 0.9999)),
            gamma=float(criterion_params.get('gamma', 2.0))
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
    ghmc_loss = ghmc_loss_fn(logits, targets)
    print("GHMCClassificationLoss:", ghmc_loss.item())

    # ==== Trường hợp lý tưởng: tất cả predict đều đúng ====
    print("\n=== Ideal case: All predictions correct ===")
    targets_perfect = torch.randint(0, num_classes, (batch_size,), device=device)
    logits_perfect = torch.full((batch_size, num_classes), -10.0, device=device, dtype=torch.float32)
    logits_perfect[torch.arange(batch_size), targets_perfect] = 10.0
    logits_perfect.requires_grad_(True)

    print("Targets (perfect):", targets_perfect)
    print(f"FocalLoss (perfect): {focal_loss(logits_perfect, targets_perfect).item():.4f}")
    print(f"F1Loss (perfect): {f1_loss(logits_perfect, targets_perfect).item():.4f}")
    print(f"CrossEntropyLoss (perfect): {ce_loss(logits_perfect, targets_perfect).item():.4f}")
    print(f"GHMCClassificationLoss (perfect): {ghmc_loss_fn(logits_perfect, targets_perfect).item():.4f}")

    # ==== Trường hợp lý tưởng: tất cả predict đều sai ====
    print("\n=== Worst case: All predictions wrong ===")
    targets_wrong = torch.randint(0, num_classes, (batch_size,), device=device)
    logits_wrong = torch.full((batch_size, num_classes), -10.0, device=device, dtype=torch.float32)
    wrong_class_indices = (targets_wrong + torch.randint(1, num_classes, (batch_size,), device=device)) % num_classes
    logits_wrong[torch.arange(batch_size), wrong_class_indices] = 10.0
    logits_wrong.requires_grad_(True)

    print("Targets (wrong):", targets_wrong)
    print(f"FocalLoss (wrong): {focal_loss(logits_wrong, targets_wrong).item():.4f}")
    print(f"F1Loss (wrong): {f1_loss(logits_wrong, targets_wrong).item():.4f}")
    print(f"CrossEntropyLoss (wrong): {ce_loss(logits_wrong, targets_wrong).item():.4f}")
    print(f"GHMCClassificationLoss (wrong): {ghmc_loss_fn(logits_wrong, targets_wrong).item():.4f}")

    # ==== Test các loss đặc biệt ====
    print("\n=== Testing IBLoss ===")
    ib_loss_fn = get_criterion('ibloss', num_classes, device)
    features = torch.rand(batch_size, device=device)
    loss_ib = ib_loss_fn(logits, targets, features)
    print("IBLoss:", loss_ib.item())

    print("\n=== Testing IB_FocalLoss ===")
    ib_focal_loss_fn = get_criterion('ib_focalloss', num_classes, device)
    loss_ib_focal = ib_focal_loss_fn(logits, targets, features)
    print("IB_FocalLoss:", loss_ib_focal.item())

    print("\n=== Testing BMCLoss ===")
    bmcloss_fn = get_criterion('bmcloss', num_classes, device)
    loss_bmc = bmcloss_fn(logits, targets)
    print("BMCLoss:", loss_bmc.item())

    print("\n=== Testing LDAMLoss ===")
    cls_num_list = [batch_size // num_classes for _ in range(num_classes)]
    ldamloss_fn = get_criterion('ldamloss', num_classes, device, {'cls_num_list': cls_num_list})
    loss_ldam = ldamloss_fn(logits, targets)
    print("LDAMLoss:", loss_ldam.item())

    print("\n=== Testing ClassBalancedLoss (CB_loss) ===")
    if num_classes == 4:
        samples_per_cls_test = [100, 50, 200, 80]
    else:
        samples_per_cls_test = [batch_size // num_classes + i*5 for i in range(num_classes)]

    print(f"Using samples_per_cls: {samples_per_cls_test} for CB_loss")

    cb_loss_focal_fn = get_criterion('cbloss', num_classes, device, 
                                     {'samples_per_cls': samples_per_cls_test, 
                                      'loss_type': 'focal', 'beta': 0.9999, 'gamma': 2.0})
    loss_cb_focal = cb_loss_focal_fn(logits, targets)
    print(f"ClassBalancedLoss (focal): {loss_cb_focal.item()}")

    cb_loss_sigmoid_fn = get_criterion('cbloss', num_classes, device, 
                                       {'samples_per_cls': samples_per_cls_test, 
                                        'loss_type': 'sigmoid', 'beta': 0.9999})
    loss_cb_sigmoid = cb_loss_sigmoid_fn(logits, targets)
    print(f"ClassBalancedLoss (sigmoid): {loss_cb_sigmoid.item()}")
    
    cb_loss_softmax_fn = get_criterion('cbloss', num_classes, device, 
                                       {'samples_per_cls': samples_per_cls_test, 
                                        'loss_type': 'softmax', 'beta': 0.9999})
    loss_cb_softmax = cb_loss_softmax_fn(logits, targets)
    print(f"ClassBalancedLoss (softmax): {loss_cb_softmax.item()}")

    print("\n=== Best case: All predictions correct (special losses) ===")
    features_perfect = torch.rand(batch_size, device=device)
    print("IBLoss (perfect):", ib_loss_fn(logits_perfect, targets_perfect, features_perfect).item())
    print("IB_FocalLoss (perfect):", ib_focal_loss_fn(logits_perfect, targets_perfect, features_perfect).item())
    print("BMCLoss (perfect):", bmcloss_fn(logits_perfect, targets_perfect).item())
    print("LDAMLoss (perfect):", ldamloss_fn(logits_perfect, targets_perfect).item())
    print(f"ClassBalancedLoss (focal, perfect): {cb_loss_focal_fn(logits_perfect, targets_perfect).item():.4f}")
    print(f"ClassBalancedLoss (sigmoid, perfect): {cb_loss_sigmoid_fn(logits_perfect, targets_perfect).item():.4f}")
    print(f"ClassBalancedLoss (softmax, perfect): {cb_loss_softmax_fn(logits_perfect, targets_perfect).item():.4f}")

    print("\n=== Worst case: All predictions wrong (special losses) ===")
    features_wrong = torch.rand(batch_size, device=device)
    print("IBLoss (wrong):", ib_loss_fn(logits_wrong, targets_wrong, features_wrong).item())
    print("IB_FocalLoss (wrong):", ib_focal_loss_fn(logits_wrong, targets_wrong, features_wrong).item())
    print("BMCLoss (wrong):", bmcloss_fn(logits_wrong, targets_wrong).item())
    print("LDAMLoss (wrong):", ldamloss_fn(logits_wrong, targets_wrong).item())
    print(f"ClassBalancedLoss (focal, wrong): {cb_loss_focal_fn(logits_wrong, targets_wrong).item():.4f}")
    print(f"ClassBalancedLoss (sigmoid, wrong): {cb_loss_sigmoid_fn(logits_wrong, targets_wrong).item():.4f}")
    print(f"ClassBalancedLoss (softmax, wrong): {cb_loss_softmax_fn(logits_wrong, targets_wrong).item():.4f}")