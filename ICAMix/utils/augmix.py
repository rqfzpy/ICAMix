import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def augmix(images, num_ops=1, severity=1, width=1, depth=1, alpha=1):
    """Apply AugMix augmentation to given images."""
    batch_size, channels, height, width = images.size()

    ws = np.float32(np.random.dirichlet([alpha] * width, size=batch_size))
    ms = np.float32(np.random.beta(alpha, alpha, size=batch_size))

    mix = torch.zeros_like(images)
    for i in range(32):
        image = images[i]
        mix_image = torch.zeros_like(image)
        for _ in range(width):
            image_aug = image.clone().to(torch.uint8)
            depth = depth if depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(augmentations)
                image_aug = op(image_aug, severity)
            mix_image += ws[i][_] * image_aug

        mix[i] = (1 - ms[i]) * image + ms[i] * mix_image

    return mix

def auto_contrast(image, _: int = 3):
    return transforms.functional.autocontrast(image)

def equalize(image, _: int = 3):
    return transforms.functional.equalize(image)

def posterize(image, severity: int = 3):
    severity = int((severity / 10) * 4)
    return transforms.functional.posterize(image, severity)

def rotate(image, severity: int = 3):
    degrees = (severity / 10) * 30
    return transforms.functional.rotate(image, degrees)

augmentations = [auto_contrast, equalize, posterize, rotate]
