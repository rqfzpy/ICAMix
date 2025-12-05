import math
import random
import numpy as np
import torch
from openmixup.models.utils import batch_shuffle_ddp, to_2tuple


@torch.no_grad()
def gridmix(img,
            gt_label,
            alpha=1.0,
            lam=None,
            n_holes=20,
            hole_aspect_ratio=1.,
            cut_area_ratio=1.,
            cut_aspect_ratio=1.,
            dist_mode=False,
            return_mask=False,
            **kwargs):
    def rand_grid(lam, size, cut_area_ratio, cut_aspect_ratio, n_holes, hole_aspect_ratio):
        W = size[2]
        H = size[3]
        cut_area = int(H * W * cut_area_ratio)
        cut_w = int(np.sqrt(cut_area / cut_aspect_ratio))
        cut_h = int(cut_w * cut_aspect_ratio)
        cx = np.random.random()
        cy = np.random.random()
        xc1 = int((W - cut_w) * cx)
        yc1 = int((H - cut_h) * cy)
        xc2 = xc1 + cut_w
        yc2 = yc1 + cut_h
        width, height = xc2 - xc1, yc2 - yc1
        patch_width = math.ceil(width / n_holes)
        patch_height = int(patch_width * hole_aspect_ratio)
        ny = math.ceil(height / patch_height)
        ratio = np.sqrt(1 - lam)
        hole_width = int(patch_width * ratio)
        hole_height = int(patch_height * ratio)
        hole_width = min(max(hole_width, 1), patch_width - 1)
        hole_height = min(max(hole_height, 1), patch_height - 1)
        holes = []
        for i in range(n_holes + 1):
            for j in range(ny + 1):
                x1 = min(patch_width * i, width)
                y1 = min(patch_height * j, height)
                x2 = min(x1 + hole_width, width)
                y2 = min(y1 + hole_height, height)
                holes.append((x1, y1, x2, y2))
        mask = torch.zeros((1, 1, W, H)).cuda()
        for x1, y1, x2, y2 in holes:
            mask[0, 0, yc1 + y1: yc1 + y2, xc1 + x1: xc1 + x2] = 1.
        return mask

    if lam is None:
        lam = np.random.beta(alpha, alpha)

    n_holes = to_2tuple(n_holes)
    hole_aspect_ratio = to_2tuple(hole_aspect_ratio)
    cut_area_ratio = to_2tuple(cut_area_ratio)
    cut_aspect_ratio = to_2tuple(cut_aspect_ratio)
    n_holes = random.randint(n_holes[0], n_holes[1])
    hole_aspect_ratio = np.random.uniform(hole_aspect_ratio[0], hole_aspect_ratio[1])
    cut_area_ratio = np.random.uniform(cut_area_ratio[0], cut_area_ratio[1])
    cut_aspect_ratio = np.random.uniform(cut_aspect_ratio[0], cut_aspect_ratio[1])

    if not dist_mode:
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:
            img_ = img[rand_index]
        else:
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        y_a = gt_label
        y_b = gt_label[rand_index]
        mask = rand_grid(lam, img.size(), cut_area_ratio, cut_aspect_ratio, n_holes, hole_aspect_ratio)
        img = img * (1 - mask) + img_ * mask
        lam = 1 - (mask[0, 0, ...].sum() / (img.shape[-1] * img.shape[-2]))
        if return_mask:
            N, _, H, W = img.size()
            img = (img, mask.expand(N, 1, H, W))
        return img, y_a, y_b, lam
    else:
        if len(img.size()) == 5:
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(img_, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        else:
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(img, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        mask = rand_grid(lam, img.size(), cut_area_ratio, cut_aspect_ratio, n_holes, hole_aspect_ratio)
        img = img * (1 - mask) + img_ * mask
        lam = 1 - (mask[0, 0, ...].sum() / (img.shape[-1] * img.shape[-2]))
        if return_mask:
            N, _, H, W = img.size()
            img = (img, mask.expand(N, 1, H, W))
        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)