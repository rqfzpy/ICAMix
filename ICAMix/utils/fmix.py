import math
import random
import torch
import numpy as np
from scipy.stats import beta
from openmixup.models.utils import batch_shuffle_ddp


def fftfreqnd(h, w=None, z=None):
    fz = fx = 0
    fy = np.fft.fftfreq(h)
    if w is not None:
        fy = np.expand_dims(fy, -1)
        fx = np.fft.fftfreq(w)[: w // 2 + 2] if w % 2 == 1 else np.fft.fftfreq(w)[: w // 2 + 1]
    if z is not None:
        fy = np.expand_dims(fy, -1)
        fz = np.fft.fftfreq(z)[:, None] if z % 2 == 1 else np.fft.fftfreq(z)[:, None]
    return np.sqrt(fx * fx + fy * fy + fz * fz)


def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    scale = np.ones(1) / (np.maximum(freqs, np.array([1.0 / max(w, h, z)])) ** decay_power)
    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)
    scale = np.expand_dims(scale, -1)[None, :]
    return scale * param


def make_low_freq_image(decay, shape, ch=1):
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))
    if len(shape) == 1:
        mask = mask[:1, : shape[0]]
    if len(shape) == 2:
        mask = mask[:1, : shape[0], : shape[1]]
    if len(shape) == 3:
        mask = mask[:1, : shape[0], : shape[1], : shape[2]]
    mask = mask - mask.min()
    mask = mask / mask.max()
    return mask


def sample_lam(alpha, reformulate=False):
    return beta.rvs(alpha + 1, alpha) if reformulate else beta.rvs(alpha, alpha)


def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)
    eff_soft = min(lam, 1 - lam) if max_soft > lam or max_soft > (1 - lam) else max_soft
    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft
    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))
    return mask.reshape((1, *in_shape))


def sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    if isinstance(shape, int):
        shape = (shape,)
    lam = sample_lam(alpha, reformulate)
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)
    return lam, mask


def sample_and_apply(x, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    index = np.random.permutation(x.shape[0])
    x1, x2 = x * mask, x[index] * (1 - mask)
    return x1 + x2, index, lam


@torch.no_grad()
def fmix(img, gt_label, alpha=1.0, lam=None, dist_mode=False, decay_power=3, size=(32, 32), max_soft=0., reformulate=False, return_mask=False, **kwargs):
    lam_, mask = sample_mask(alpha, decay_power, size, max_soft, reformulate)
    mask = torch.from_numpy(mask).cuda().type_as(img)
    if lam is None:
        lam = lam_
    else:
        if lam_ < lam:
            mask = 1 - mask
            lam = 1 - lam_
    if not dist_mode:
        indices = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:
            img_ = img[indices]
        else:
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        y_a = gt_label
        y_b = gt_label[indices]
        img = mask * img + (1 - mask) * img_
        if return_mask:
            N, _, H, W = img.shape
            img = (img, mask.expand(N, 1, H, W))
        return img, y_a, y_b, lam
    else:
        if len(img.size()) == 5:
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(img_, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        else:
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(img, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        img = mask * img + (1 - mask) * img_
        if return_mask:
            N, _, H, W = img.shape
            img = (img, mask.expand(N, 1, H, W))
        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, idx_shuffle, idx_unshuffle, lam