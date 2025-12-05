import numpy as np
import utils.pixmix_utils as utils
import torch

def augment_input(image):
    aug_list = utils.augmentations_all
    op = np.random.choice(aug_list)
    return op(image.copy(), 1)

def pixmix(orig, mixing_pic, preprocess):
    mixings = utils.mixings
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig))
    else:
        mixed = tensorize(orig)
    for _ in range(np.random.randint(4 + 1)):
        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(orig))
        else:
            aug_image_copy = tensorize(mixing_pic)
        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, 4)
        mixed = torch.clip(mixed, 0, 1)
    return normalize(mixed)

class PixMixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mixing_set, preprocess):
        self.dataset = dataset
        self.mixing_set = mixing_set
        self.preprocess = preprocess

    def __getitem__(self, i):
        x, y = self.dataset[i]
        rnd_idx = np.random.choice(len(self.mixing_set))
        mixing_pic, _ = self.mixing_set[rnd_idx]
        return pixmix(x, mixing_pic, self.preprocess), y

    def __len__(self):
        return len(self.dataset)