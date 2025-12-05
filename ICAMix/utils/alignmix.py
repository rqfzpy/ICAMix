import numpy as np
import random
import torch
import torch.nn as nn


class SinkhornDistance(nn.Module):
    def __init__(self, eps=0.1, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        C = self._cost_matrix(x, y)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        batch_size = 1 if x.dim() == 2 else x.shape[0]
        mu = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()
        u = torch.zeros_like(mu).cuda()
        v = torch.zeros_like(nu).cuda()
        actual_nits = 0
        thresh = 1e-1
        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self._M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self._M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            actual_nits += 1
            if err.item() < thresh:
                break
        U, V = u, v
        pi = torch.exp(self._M(C, U, V))
        return pi

    def _M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        x_col = x.unsqueeze(-2).cuda()
        y_lin = y.unsqueeze(-3).cuda()
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        return tau * u + (1 - tau) * u1


def mixup_aligned(feat1, feat2, lam, eps=0.1, max_iter=100):
    B, C, H, W = feat1.shape
    feat1 = feat1.view(B, C, -1)
    feat2 = feat2.view(B, C, -1)
    sinkhorn = SinkhornDistance(eps=eps, max_iter=max_iter, reduction=None)
    P = sinkhorn(feat1.permute(0, 2, 1), feat2.permute(0, 2, 1)).detach()
    P = P * (H * W)
    align_mix = random.randint(0, 1)
    if (align_mix == 0):
        f1 = torch.matmul(feat2, P.permute(0, 2, 1).cuda()).view(B, C, H, W)
        final = feat1.view(B, C, H, W) * lam + f1 * (1 - lam)
    elif (align_mix == 1):
        f2 = torch.matmul(feat1, P.cuda()).view(B, C, H, W).cuda()
        final = f2 * lam + feat2.view(B, C, H, W) * (1 - lam)
    return final


def alignmix(img, gt_label, alpha=1.0, lam=None, dist_mode=False, eps=0.1, max_iter=100, **kwargs):
    if lam is None:
        lam = np.random.beta(alpha, alpha)
    if not dist_mode:
        rand_index = torch.randperm(img.size(0))[0].cuda()
        if len(img.size()) == 4:
            img_ = img[rand_index]
        else:
            assert img.dim() == 5
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        y_a = gt_label
        y_b = gt_label[rand_index]
        feat = mixup_aligned(img, img_, lam, eps, max_iter)
        return feat, y_a, y_b, lam
    else:
        raise ValueError("AlignMix cannot perform distributed mixup.")