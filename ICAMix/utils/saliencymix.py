import numpy as np
import torch

try:
    from cv2.saliency import StaticSaliencyFineGrained_create
except ImportError:
    StaticSaliencyFineGrained_create = None
from openmixup.models.utils import batch_shuffle_ddp


@torch.no_grad()
def saliencymix(img,
                gt_label,
                alpha=1.0,
                lam=None,
                dist_mode=False,
                return_mask=False,
                **kwargs):
    if StaticSaliencyFineGrained_create is None:
        raise RuntimeError(
            'Failed to import `StaticSaliencyFineGrained_create` from cv2 for SaliencyMix. '
            'Please install it by "pip install opencv-contrib-python".')

    def saliency_bbox(img, lam):
        size = img.size()
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        img = img.type(torch.float32)
        temp_img = img.cpu().numpy().transpose(1, 2, 0)
        saliency = StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        maximum_indices = np.unravel_index(
            np.argmax(saliencyMap, axis=None), saliencyMap.shape)
        x = maximum_indices[0]
        y = maximum_indices[1]
        bbx1 = np.clip(x - cut_w // 2, 0, W)
        bby1 = np.clip(y - cut_h // 2, 0, H)
        bbx2 = np.clip(x + cut_w // 2, 0, W)
        bby2 = np.clip(y + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

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
        bbx1, bby1, bbx2, bby2 = saliency_bbox(img[rand_index[0]], lam)
        img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        if return_mask:
            mask = torch.zeros((1, 1, h, w)).cuda()
            mask[:, :, bbx1:bbx2, bby1:bby2] = 1
            mask = mask.expand(b, 1, h, w)
            img = (img, mask)
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
        bbx1, bby1, bbx2, bby2 = saliency_bbox(img_[0], lam)
        img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        if return_mask:
            mask = torch.zeros((1, 1, h, w)).cuda()
            mask[:, :, bbx1:bbx2, bby1:bby2] = 1
            mask = mask.expand(b, 1, h, w)
            img = (img, mask)
        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)