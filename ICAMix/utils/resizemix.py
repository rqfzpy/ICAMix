import numpy as np
import torch
from torch.nn.functional import interpolate
from openmixup.models.utils import batch_shuffle_ddp


@torch.no_grad()
def resizemix(img,
              gt_label,
              scope=(0.1, 0.8),
              dist_mode=False,
              alpha=1.0,
              lam=None,
              use_alpha=False,
              interpolate_mode="nearest",
              return_mask=False,
              **kwargs):
    def rand_bbox_tao(size, tao, return_mask=False):
        W = size[2]
        H = size[3]
        cut_w = int(W * tao)
        cut_h = int(H * tao)
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

    assert len(scope) == 2

    if not dist_mode:
        rand_index = torch.randperm(img.size(0))
        if len(img.size()) == 4:
            img_resize = img.clone()
            img_resize = img_resize[rand_index]
        else:
            img_resize = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        _, _, h, w = img.size()
        shuffled_gt = gt_label[rand_index]
        if lam is None:
            if use_alpha:
                tao = np.random.beta(alpha, alpha)
                if tao < scope[0] or tao > scope[1]:
                    tao = np.random.uniform(scope[0], scope[1])
            else:
                tao = np.random.uniform(scope[0], scope[1])
        else:
            tao = min(max(lam, scope[0]), scope[1])
        if not return_mask:
            bbx1, bby1, bbx2, bby2 = rand_bbox_tao(img.size(), tao)
        else:
            bbx1, bby1, bbx2, bby2, mask = rand_bbox_tao(img.size(), tao, True)
        img_resize = interpolate(
            img_resize, (bby2 - bby1, bbx2 - bbx1), mode=interpolate_mode
        )
        img[:, :, bby1:bby2, bbx1:bbx2] = img_resize
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        if return_mask:
            img = (img, mask)
        return img, (gt_label, shuffled_gt, lam)
    else:
        if len(img.size()) == 5:
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(
                img_, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        else:
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(
                img, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        _, _, h, w = img.size()
        if lam is None:
            if use_alpha:
                tao = np.random.beta(alpha, alpha)
                if tao < scope[0] or tao > scope[1]:
                    tao = np.random.uniform(scope[0], scope[1])
            else:
                tao = np.random.uniform(scope[0], scope[1])
        else:
            tao = lam
        if not return_mask:
            bbx1, bby1, bbx2, bby2 = rand_bbox_tao(img.size(), tao)
        else:
            bbx1, bby1, bbx2, bby2, mask = rand_bbox_tao(img.size(), tao, True)
        img_ = interpolate(img_, (bby2 - bby1, bbx2 - bbx1), mode=interpolate_mode)
        img[:, :, bby1:bby2, bbx1:bbx2] = img_
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        if return_mask:
            img = (img, mask)
        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)