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
    def __init__(self, num_classes, beta=1.0, reduction='mean'):
        super(F1Loss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        preds = torch.argmax(inputs, dim=1)

        tp = (preds * targets).sum().to(torch.float32)
        fp = ((1 - targets) * preds).sum().to(torch.float32)
        fn = (targets * (1 - preds)).sum().to(torch.float32)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        f1_score = (1 + self.beta ** 2) * (precision * recall) / (self.beta ** 2 * precision + recall + 1e-6)

        if self.reduction == 'mean':
            return 1 - f1_score.mean()
        elif self.reduction == 'sum':
            return 1 - f1_score.sum()
        return 1 - f1_score