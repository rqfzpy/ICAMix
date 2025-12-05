import numpy as np
import torch
from openmixup.models.utils import batch_shuffle_ddp


@torch.no_grad()
def smoothmix(img,
              gt_label,
              alpha=1.0,
              lam=None,
              dist_mode=False,
              return_mask=False,
              **kwargs):
    def gaussian_kernel(kernel_size, rand_w, rand_h, sigma):
        s = kernel_size * 2
        x_cord = torch.arange(s)
        x_grid = x_cord.repeat(s).view(s, s)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).cuda()
        xy_grid = torch.roll(xy_grid, rand_w, 0)
        xy_grid = torch.roll(xy_grid, rand_h, 1)
        crop_size = s // 4
        xy_grid = xy_grid[crop_size: s - crop_size, crop_size: s - crop_size]
        mean = (s - 1) / 2
        var = sigma ** 2
        g_filter = torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * var))
        g_filter = g_filter.view(kernel_size, kernel_size)
        return g_filter

    if lam is None:
        lam = np.random.beta(alpha, alpha)

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
        rand_w = int(torch.randint(0, w, (1,)) - w / 2)
        rand_h = int(torch.randint(0, h, (1,)) - h / 2)
        sigma = ((torch.rand(1) / 4 + 0.25) * h).cuda()
        kernel = gaussian_kernel(h, rand_h, rand_w, sigma).cuda()
        img = img * (1 - kernel) + img_ * kernel
        lam = torch.sum(kernel) / (h * w)
        if return_mask:
            img = (img, kernel.expand(b, 1, h, w))
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
        rand_w = int(torch.randint(0, w, (1,)) - w / 2)
        rand_h = int(torch.randint(0, h, (1,)) - h / 2)
        sigma = (torch.rand(1) / 4 + 0.25) * h
        kernel = gaussian_kernel(h, rand_h, rand_w, sigma).cuda()
        img = img * (1 - kernel) + img_ * kernel
        lam = torch.sum(kernel) / (h * w)
        if return_mask:
            img = (img, kernel.expand(b, 1, h, w))
        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)