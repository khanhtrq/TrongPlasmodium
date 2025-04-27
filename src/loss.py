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
        self.eps = 1e-6  # Small epsilon to prevent numerical instability

    def forward(self, inputs, targets):
        # Apply log_softmax for numerical stability
        logpt = F.log_softmax(inputs, dim=1)
        
        # Get pt = exp(logpt)
        pt = torch.exp(logpt)
        
        # Select the appropriate values for the target class
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply alpha weighting if provided
        if isinstance(self.alpha, torch.Tensor):
            try:
                at = self.alpha.gather(0, targets)
            except:
                at = torch.tensor(1.0, device=inputs.device)
        else:
            at = torch.tensor(self.alpha, device=inputs.device)
    
        # Calculate focal loss with numerical stability safeguards
        # Use clipping to prevent extreme values
        pt_safe = torch.clamp(pt, min=self.eps, max=1.0)
        focal_weight = at * (1 - pt_safe).pow(self.gamma)
        
        # Final loss calculation
        loss = -focal_weight * logpt
    
        # Apply reduction as specified
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def compute_alpha_from_dataloader(train_loader, num_classes, device):
    """Compute class weights from a dataloader for focal loss alpha parameter."""
    print("Computing class weights for balanced focal loss...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())
    
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for lbl in all_labels:
        class_counts[lbl] += 1
    
    # Compute inverse frequency weights
    total_samples = class_counts.sum()
    class_weights = total_samples / (class_counts * num_classes + 1e-6)
    
    # Normalize weights to sum to num_classes
    class_weights = class_weights * (num_classes / class_weights.sum())
    
    print(f"Class weights: {class_weights.tolist()}")
    
    return class_weights.to(device)

def compute_class_weights(dataloader, num_classes, device):
    """Alternative method to compute class weights."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in dataloader:
        for label in labels:
            counts[label] += 1
    
    # Avoid division by zero
    counts = torch.clamp(counts, min=1.0)
    
    # Calculate inverse frequency weights
    total = counts.sum()
    weights = total / (counts * num_classes)
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    print(f"Computed class weights: {weights.tolist()}")
    
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
        # Apply softmax for numerical stability
        probas = F.softmax(inputs, dim=1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).to(dtype=inputs.dtype)

        # Calculate true positives, false positives, false negatives per class
        tp = (probas * targets_one_hot).sum(dim=0)
        fp = (probas * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - probas) * targets_one_hot).sum(dim=0)

        # Calculate precision and recall per class
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        # Calculate F1 score per class
        f1 = (1 + self.beta**2) * (precision * recall) / ((self.beta**2 * precision) + recall + self.epsilon)

        # Average F1 score across classes (macro F1)
        f1_loss = 1 - f1.mean()

        return f1_loss