import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomCrossEntropyLoss:
    def __init__(self):
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
    
    def __call__(self, output, target):
        return self.compute_loss(output, target)
    
    def compute_loss(self, output, target):
        return torch.mean(torch.sum(-target * self.logsoftmax(output), dim=1))

def smooth_one_hot(target: torch.Tensor, num_classes: int, smoothing=0.0):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    true_dist = target.new_zeros(size=(len(target), num_classes)).float()
    true_dist.fill_(smoothing / (num_classes - 1))
    true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
    return true_dist

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing < 1
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        bs = float(pred.size(0))
        pred = pred.log_softmax(dim=1)
        if len(target.shape) == 2:
            true_dist = target
        else:
            true_dist = smooth_one_hot(target, self.num_classes, self.smoothing)
        loss = (-pred * true_dist).sum() / bs
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def _compute_losses(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        return self._compute_losses(x, target).mean()

class AbsLoss(object):
    def __init__(self):
        self.record = []
        self.bs = []
    
    def compute_loss(self, pred, gt):
        pass
    
    def _update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss
    
    def _average_loss(self):
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record * bs).sum() / bs.sum()
    
    def _reinit(self):
        self.record = []
        self.bs = []

class KL_DivLoss(AbsLoss):
    def __init__(self):
        super(KL_DivLoss, self).__init__()
        self.loss_fn = nn.KLDivLoss()
        
    def compute_loss(self, pred, gt):
        loss = self.loss_fn(pred, gt)
        return loss

class JSloss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSloss, self).__init__()
        self.red = reduction
        
    def forward(self, input, target):
        net = F.softmax(((input + target) / 2.), dim=1)
        return 0.5 * (F.kl_div(input.log(), net, reduction=self.red) + 
                      F.kl_div(target.log(), net, reduction=self.red))

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, p_pred: torch.Tensor, p_true: torch.Tensor):
        assert p_true.shape == p_pred.shape
        cdf_target = torch.cumsum(p_true, dim=1)
        cdf_estimate = torch.cumsum(p_pred, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=1))
        return samplewise_emd.mean()