import numpy as np
import torch
import random
import torch.nn.functional as F
from scipy.fftpack import dct, idct
import pywt
from pytorch_wavelets import DWTForward, DWTInverse


def ICAMix(x, y, num_class, lamb, dylamb):
    img_magnitudemix = torch.zeros_like(x)
    
    for i in range(num_class):
        class_data = x[y == i]
        indices = torch.where(y == i)
        
        if class_data.shape[0] == 0:
            continue
        
        fft_feature = torch.fft.rfftn(class_data, dim=[2, 3])
        fft_pha = torch.angle(fft_feature)
        mixed_fft_amp = torch.mean(torch.abs(fft_feature), dim=0, keepdim=True).repeat(fft_feature.shape[0], 1, 1, 1)
        mixed_fft_feature = mixed_fft_amp * torch.exp(1j * fft_pha)
        mixed_data = torch.fft.irfftn(mixed_fft_feature, s=(x.shape[2], x.shape[3]), dim=[2, 3])
        mixed_data = (mixed_data - mixed_data.min()) / (mixed_data.max() - mixed_data.min() + 1e-8) 
        mixed_data = mixed_data * 255 
        img_magnitudemix[indices] = mixed_data  
    
    mixed_x = (dylamb - lamb) * x + lamb * img_magnitudemix
    return mixed_x
