import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', eps=1e-8):
        super(FocalLoss, self).__init__()
        
        if gamma < 0:
            # While the question is about 0 < gamma < 1, ensuring gamma is non-negative is good practice.
            raise ValueError("Gamma must be non-negative for FocalLoss.")
        
        if eps <= 0:
            raise ValueError("Epsilon (eps) must be a positive value for numerical stability.")
            
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps  # A small epsilon for numerical stability

    def forward(self, inputs, targets):
        # inputs: (batch_size, num_classes) - Raw logits
        # targets: (batch_size) - Long tensor of class indices

        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        
        # Gather log probabilities and probabilities for the target class
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Clamp pt to prevent log(0) and (0)^x issues, crucial for stability.
        # This clamping ensures (1-pt) is also kept away from 0.
        # For 0 < gamma < 1, (1-pt)^(gamma-1) appears in the gradient.
        # If 1-pt were 0, and gamma-1 < 0, this would lead to division by zero (infinity).
        pt = pt.clamp(min=self.eps, max=1.0 - self.eps)

        # Calculate the focal loss term
        # (1-pt)^gamma
        focal_term = (1.0 - pt).pow(self.gamma)

        # Alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Assumes alpha is a 1D tensor of weights for each class
                at = self.alpha.gather(0, targets)
            elif isinstance(self.alpha, (float, int)):
                at = self.alpha
            else:
                # Fallback if alpha is an unexpected type, though ideally, type checking in __init__ would be stricter.
                at = 1.0 
        else:
            at = 1.0

        loss = -at * focal_term * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        # if 'none' or other, return element-wise loss
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
    
def get_criterion(criterion, num_classes, device, criterion_params=None):
    criterion = criterion.lower() if isinstance(criterion, str) else criterion
    criterion_params = criterion_params or {}
    print(f"Using criterion: {criterion} with params: {criterion_params}")
    if criterion == 'focalloss':
        # Allow passing alpha, gamma, reduction
        return FocalLoss(
            alpha=criterion_params.get('alpha', 1.0),
            gamma=criterion_params.get('gamma', 2.0),
            reduction=criterion_params.get('reduction', 'mean')
        ).to(device)
    elif criterion == 'f1loss':
        # Allow passing beta, epsilon, reduction
        return F1Loss(
            num_classes=num_classes,
            epsilon=criterion_params.get('epsilon', 1e-7),
            beta=criterion_params.get('beta', 1.0),
            reduction=criterion_params.get('reduction', 'mean')
        ).to(device)
    else:
        # For CrossEntropyLoss, allow passing weight, reduction, etc.
        ce_kwargs = {}
        if 'weight' in criterion_params:
            ce_kwargs['weight'] = criterion_params['weight']
        if 'reduction' in criterion_params:
            ce_kwargs['reduction'] = criterion_params['reduction']
        return nn.CrossEntropyLoss(**ce_kwargs).to(device)