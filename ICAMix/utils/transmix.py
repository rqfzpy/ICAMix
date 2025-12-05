import numpy as np
import torch
import torch.nn as nn
from openmixup.models.utils import batch_shuffle_ddp


@torch.no_grad()
def transmix(img,
             gt_label,
             dist_mode=False,
             alpha=1.0,
             mask=None,
             lam=None,
             attn=None,
             patch_shape=None,
             return_mask=False,
             ratio=0.5,
             **kwargs):
    def rand_bbox(size, lam, return_mask=False):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        if not return_mask:
            return bbx1, bby1, bbx2, bby2
        else:
            mask = torch.zeros((1, 1, W, H)).cuda()
            mask[:, :, bbx1:bbx2, bby1:bby2] = 1
            mask = mask.expand(size[0], 1, W, H)
            return bbx1, bby1, bbx2, bby2, mask

    if lam is None and mask is None:
        lam0 = np.random.beta(alpha, alpha)

    if not dist_mode:
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:
            img_ = img[rand_index]
        else:
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        b, _, h, w = img.size()
        y_a = gt_label
        y_b = gt_label[rand_index]
        if mask is None:
            bbx1, bby1, bbx2, bby2, mask = rand_bbox(img.size(), lam0, True)
            img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
            lam0 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        else:
            img = (1-mask) * img + mask * img_
            lam0 = torch.mean(mask[0, 0, ...]) / (h * w) if lam is None else lam
        if return_mask:
            img = (img, mask)
        lam1 = lam0
        if attn is not None:
            mask_ = nn.Upsample(size=patch_shape)(mask).view(b, -1).int()
            attn_ = torch.mean(attn[:, :, 0, 1:], dim=1)
            w1, w2 = torch.sum(mask_ * attn_, dim=1), torch.sum((1-mask_) * attn_, dim=1)
            lam1 = w2 / (w1+w2)
        lam = lam0 * ratio + lam1 * (1-ratio)
        return img, (y_a, y_b, lam)
    else:
        if len(img.size()) == 5:
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(
                img_, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        else:
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(
                img, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        b, _, h, w = img.size()
        if mask is None:
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam0)
            img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
            lam0 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        else:
            img = (1-mask) * img + mask * img_
            lam0 = torch.mean(mask[0, 0, ...]) / (h * w) if lam is None else lam
        if return_mask:
            img = (img, mask)
        lam1 = lam0
        if attn is not None:
            mask_ = nn.Upsample(size=patch_shape)(mask).view(b, -1).int()
            attn_ = torch.mean(attn[:, :, 0, 1:], dim=1)
            w1, w2 = torch.sum((1-mask_) * attn_, dim=1), torch.sum(mask_ * attn_, dim=1)
            lam1 = w2 / (w1+w2)
        lam = lam0 * ratio + lam1 * (1-ratio)
        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)