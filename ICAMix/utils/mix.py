import numpy as np
import torch
import random
import torch.nn.functional as F
from scipy.fftpack import dct, idct
import pywt
from pytorch_wavelets import DWTForward, DWTInverse




###############################################
# 实现 2D DCT / IDCT（支持 GPU）
###############################################
def dct_2d(x):
    # x: (B, C, H, W)
    return torch.real(torch.fft.fft2(x, norm="ortho"))


def idct_2d(X):
    # X: (B, C, H, W)
    return torch.real(torch.fft.ifft2(X, norm="ortho"))


###############################################
# GPU Accelerated DCT Magnitude Mix
###############################################
def MagnitudeMix_DCT_bf1(x, y, num_class, lamb, dylamb):
    """
    x: B×C×H×W 图像（RGB/Hyperspectral 都可以）
    y: B 标签
    num_class: 类别数
    lamb, dylamb: mix 参数
    """
    device = x.device
    B, C, H, W = x.shape
    out = torch.zeros_like(x)

    for cls in range(num_class):
        idx = (y == cls).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue

        # Extract class samples
        cls_x = x[idx]  # (Nc, C, H, W)

        # Step 1 — DCT 变换
        X_dct = dct_2d(cls_x)  # Nc × C × H × W

        # Step 2 — 幅度 + 相位
        X_amp = torch.abs(X_dct)
        X_phase = torch.sign(X_dct)  # DCT 可以用符号近似相位

        # Step 3 — 类内幅度平均（Magnitude Mix）
        mean_amp = X_amp.mean(dim=0, keepdim=True).repeat(X_amp.size(0), 1, 1, 1)  # Nc×C×H×W

        mixed_amp = lamb * mean_amp + (dylamb - lamb) * X_amp

        # Step 4 — 重建 DCT 系数
        mixed_dct = mixed_amp * X_phase

        # Step 5 — 逆 DCT
        mixed_img = idct_2d(mixed_dct)

        out[idx] = mixed_img

    return out
###############################################
# GPU Accelerated DCT Magnitude Mix
###############################################
def MagnitudeMix_DCT(x, y, num_class, lamb, dylamb):
    """
    y: B 标签
    num_class: 类别数
    lamb, dylamb: mix 参数
    """
    device = x.device
    B, C, H, W = x.shape
    out = torch.zeros_like(x)

    for cls in range(num_class):
        idx = (y == cls).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue

        cls_x = x[idx]  # Nc × C × H × W

        # Step 1 — DCT 变换
        X_dct = torch.fft.fft2(cls_x, norm='ortho')

        # Step 2 — 幅度 + 相位
        X_amp = torch.abs(X_dct)
        X_phase = torch.exp(1j * torch.angle(X_dct))

        # Step 3 — 类内幅度平均（Magnitude Mix）
        mean_amp = X_amp.mean(dim=0, keepdim=True).repeat(X_amp.size(0), 1, 1, 1)

        mixed_amp = lamb * mean_amp + (dylamb - lamb) * X_amp

        # Step 4 — 重建 DCT 系数
        mixed_dct = mixed_amp * X_phase

        # Step 5 — 逆 DCT
        mixed_img = torch.fft.ifft2(mixed_dct, norm='ortho').real

        out[idx] = mixed_img

    return out


###############################################
# GPU Accelerated DWT Magnitude Mix
###############################################
def MagnitudeMix_DWT(x, y, num_class, lamb, dylamb, wave='haar', J=1):
    """
    x: 输入图像 B×C×H×W
    y: 标签 B
    num_class: 类别数量
    lamb, dylamb: mix 参数
    wave: 小波类型(haar/db1 推荐)
    J: 小波层数，1 层最稳定
    """
    device = x.device
    B, C, H, W = x.shape
    out = torch.zeros_like(x)

    # 自动填充到偶数大小，保证 DWT 正确
    padH = H % 2
    padW = W % 2
    if padH != 0 or padW != 0:
        x = F.pad(x, (0, padW, 0, padH), mode='reflect')

    # 定义 forward 和 inverse wavelet
    dwt = DWTForward(J=J, wave=wave).to(device)
    idwt = DWTInverse(wave=wave).to(device)

    for cls in range(num_class):
        idx = (y == cls).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue

        cls_x = x[idx]  # Nc × C × H × W

        # Step1 — 小波变换
        LL, highs = dwt(cls_x)  # LL: Nc×C×H/2×W/2, highs: list

        # Step2 — 低频幅度平均 (Magnitude Mix)
        LL_mix = LL.abs().mean(dim=0, keepdim=True).repeat(LL.size(0), 1, 1, 1)

        # Step3 — 混合
        mixed_LL = lamb * LL_mix + (dylamb - lamb) * LL

        # 高频保持不变
        mixed_highs = [h for h in highs]

        # Step4 — 逆小波
        mixed_img = idwt((mixed_LL, mixed_highs))

        # 去掉填充
        if padH != 0 or padW != 0:
            mixed_img = mixed_img[:, :, :H, :W]

        out[idx] = mixed_img

    return out


def dwt2_batch_rgb(x, wavelet='haar'):
    """
    x: B×3×H×W 的 RGB 图像
    返回：
        LL, LH, HL, HH --> 每个都是 B×3×H/2×W/2
    """
    B, C, H, W = x.shape
    LL_list, LH_list, HL_list, HH_list = [], [], [], []

    for b in range(B):
        LL_c, LH_c, HL_c, HH_c = [], [], [], []
        for c in range(C):
            img_c = x[b, c].cpu().numpy()
            LL, (LH, HL, HH) = pywt.dwt2(img_c, wavelet)
            LL_c.append(LL)
            LH_c.append(LH)
            HL_c.append(HL)
            HH_c.append(HH)

        # 合并通道
        LL_list.append(torch.tensor(np.stack(LL_c)))
        LH_list.append(torch.tensor(np.stack(LH_c)))
        HL_list.append(torch.tensor(np.stack(HL_c)))
        HH_list.append(torch.tensor(np.stack(HH_c)))

    return (
        torch.stack(LL_list).to(x.device),
        torch.stack(LH_list).to(x.device),
        torch.stack(HL_list).to(x.device),
        torch.stack(HH_list).to(x.device)
    )

def idwt2_batch_rgb(LL, LH, HL, HH, wavelet='haar'):
    B, C, H2, W2 = LL.shape
    out = []

    for b in range(B):
        ch_list = []
        for c in range(C):
            coeffs = (LL[b, c].cpu().numpy(),
                     (LH[b, c].cpu().numpy(),
                      HL[b, c].cpu().numpy(),
                      HH[b, c].cpu().numpy()))
            rec = pywt.idwt2(coeffs, wavelet)
            ch_list.append(torch.tensor(rec))
        out.append(torch.stack(ch_list))

    return torch.stack(out).to(LL.device)

def dct2(a):
    # 二维 DCT-II
    return dct(dct(a, axis=2, norm='ortho'), axis=3, norm='ortho')

def idct2(a):
    # 二维逆 DCT-II
    return idct(idct(a, axis=2, norm='ortho'), axis=3, norm='ortho')

def dwt2_batch(x, wavelet='haar'):
    """对批量图像进行 DWT 分解，返回 (LL, (LH, HL, HH))"""
    coeffs = [pywt.dwt2(img.cpu().numpy(), wavelet) for img in x]
    LL, (LH, HL, HH) = zip(*coeffs)
    return (torch.tensor(LL), torch.tensor(LH), torch.tensor(HL), torch.tensor(HH))

def idwt2_batch(LL, LH, HL, HH, wavelet='haar'):
    """DWT 逆变换"""
    rec = [pywt.idwt2((LL[i].numpy(), (LH[i].numpy(), HL[i].numpy(), HH[i].numpy())), wavelet)
           for i in range(len(LL))]
    return torch.tensor(rec)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, args):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if args.alpha > 0:
        lam = np.random.beta(args.alpha, args.alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam




def same_class_mixup_data(x, y,num_class,args):
        # Initialize an empty tensor for mixed images
        img_magnitudemix = torch.zeros_like(x)

        if args.alpha > 0:
            lam = np.random.beta(args.alpha, args.alpha)
        else:
            lam = 1
        
        # Iterate over each class
        for i in range(num_class):
            # Extract data corresponding to the current class
            class_data = x[y == i]
            indices = torch.where(y == i)[0]
            # 打乱索引顺序
            shuffled_indices = torch.randperm(indices.size(0))
            shuffled_indices = indices[shuffled_indices]
            shuffled_class_data = x[shuffled_indices]

            mixed_x = lam * class_data + (1 - lam) * shuffled_class_data

            img_magnitudemix[indices] = mixed_x 
        
        
        # Return the mixed image
        return img_magnitudemix


def cutmix_data(x, y, args):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if args.beta > 0:
        lam = np.random.beta(args.beta, args.beta)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).cuda()

    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_sliced = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby1 - bby2) / (x.size()[-1] * x.size()[-2]))
    
    return [bbx1, bby1, bbx2, bby2 ], y_a, y_b, lam, x_sliced

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def MagnitudeMix(x, y,num_class,lamb,dylamb):
        # Initialize an empty tensor for mixed images
        img_magnitudemix = torch.zeros_like(x)
        
        # Iterate over each class
        for i in range(num_class):
            # Extract data corresponding to the current class
            class_data = x[y == i]
            
            # Find indices where the labels match the current class
            indices = torch.where(y == i)
            
            # Skip if no data is found for the current class
            if class_data.shape[0] == 0:
                continue
            
            # Compute the FFT of the class data
            fft_feature = torch.fft.rfftn(class_data, dim=[2, 3])
            
            # Compute the phase of the FFT
            fft_pha = torch.angle(fft_feature)
            
            # Compute the mixed FFT amplitude
            mixed_fft_amp = torch.mean(torch.abs(fft_feature), dim=0, keepdim=True).repeat(fft_feature.shape[0], 1, 1, 1)
            
            # Combine the mixed FFT amplitude with the original phase
            mixed_fft_feature = mixed_fft_amp * torch.exp(1j * fft_pha)
            
            # Compute the inverse FFT to obtain the mixed data
            mixed_data = torch.fft.irfftn(mixed_fft_feature, s=(x.shape[2], x.shape[3]), dim=[2, 3])
            
            # Update the mixed image tensor with the mixed data for the current class
            img_magnitudemix[indices] = mixed_data 
        
        # Add the original image multiplied by lambda to the mixed image
        mixed_x = (dylamb - lamb)* x + lamb * img_magnitudemix
        
        # Return the mixed image
        return mixed_x

def MagnitudeMix_DCT_cpu(x, y, num_class, lamb, dylamb):
    img_magnitudemix = torch.zeros_like(x)
    
    for i in range(num_class):
        class_data = x[y == i]
        indices = torch.where(y == i)
        if class_data.shape[0] == 0:
            continue
        
        # 转为numpy计算DCT
        data_np = class_data.cpu().numpy()
        dct_feature = dct2(data_np)
        
        # 混合 DCT 幅值
        mixed_dct_amp = abs(dct_feature).mean(axis=0, keepdims=True)
        mixed_dct_feature = mixed_dct_amp * (dct_feature / (abs(dct_feature) + 1e-8))
        
        # 逆 DCT 回图像
        mixed_data = idct2(mixed_dct_feature).real
        mixed_data = torch.from_numpy(mixed_data).to(x.device, dtype=x.dtype)
        
        img_magnitudemix[indices] = mixed_data
    
    mixed_x = (dylamb - lamb) * x + lamb * img_magnitudemix
    return mixed_x


def MagnitudeMix_DWT_cpu(x, y, num_class, lamb, dylamb, wavelet='haar'):
    img_magnitudemix = torch.zeros_like(x)

    for i in range(num_class):
        class_data = x[y == i]
        indices = torch.where(y == i)

        if class_data.shape[0] == 0:
            continue

        # --- DWT ---
        LL, LH, HL, HH = dwt2_batch_rgb(class_data, wavelet=wavelet)

        # --- 计算幅度(取平均操作类似 DFT 版本) ---
        LL_mix = LL.abs().mean(dim=0, keepdim=True).repeat(LL.shape[0], 1, 1, 1)
        LH_mix = LH.abs().mean(dim=0, keepdim=True).repeat(LH.shape[0], 1, 1, 1)
        HL_mix = HL.abs().mean(dim=0, keepdim=True).repeat(HL.shape[0], 1, 1, 1)
        HH_mix = HH.abs().mean(dim=0, keepdim=True).repeat(HH.shape[0], 1, 1, 1)

        # --- IDWT ---
        mixed_data = idwt2_batch_rgb(LL_mix, LH_mix, HL_mix, HH_mix, wavelet)

        img_magnitudemix[indices] = mixed_data

    mixed_x = (dylamb - lamb) * x + lamb * img_magnitudemix
    return mixed_x


def MagALLMix(x, y,num_class,lamb,dylamb):
    
        # Compute the FFT of the class data
        fft_feature = torch.fft.rfftn(x, dim=[2, 3])
        
        # Compute the phase of the FFT
        fft_pha = torch.angle(fft_feature)
        
        # Compute the mixed FFT amplitude
        mixed_fft_amp = torch.mean(torch.abs(fft_feature), dim=0, keepdim=True).repeat(fft_feature.shape[0], 1, 1, 1)
        
        # Combine the mixed FFT amplitude with the original phase
        mixed_fft_feature = mixed_fft_amp * torch.exp(1j * fft_pha)
        
        # Compute the inverse FFT to obtain the mixed data
        mixed_data = torch.fft.irfftn(mixed_fft_feature, s=(x.shape[2], x.shape[3]), dim=[2, 3])
        
        # Update the mixed image tensor with the mixed data for the current class
        
        # Add the original image multiplied by lambda to the mixed image
        mixed_x = (dylamb - lamb)* x + lamb * mixed_data
        
        # Return the mixed image
        return mixed_x

def MagRandomMix(x, y,num_class,lamb,dylamb):

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        random_x = x[index, :]

        # Compute the FFT of the class data
        fft_feature = torch.fft.rfftn(random_x, dim=[2, 3])
        
        # Compute the phase of the FFT
        fft_pha = torch.angle(fft_feature)
        
        # Compute the mixed FFT amplitude
        mixed_fft_amp = torch.mean(torch.abs(fft_feature), dim=0, keepdim=True).repeat(fft_feature.shape[0], 1, 1, 1)
        
        # Combine the mixed FFT amplitude with the original phase
        mixed_fft_feature = mixed_fft_amp * torch.exp(1j * fft_pha)
        
        # Compute the inverse FFT to obtain the mixed data
        mixed_data = torch.fft.irfftn(mixed_fft_feature, s=(x.shape[2], x.shape[3]), dim=[2, 3])
        
        # Update the mixed image tensor with the mixed data for the current class
        
        # Add the original image multiplied by lambda to the mixed image
        mixed_x = (dylamb - lamb)* x + lamb * mixed_data
        
        # Return the mixed image
        return mixed_x

def cutout(img, n_holes=1, length=16):
    """
    在输入图像上应用 Cutout 数据增强。

    参数：
    - img: 输入图像张量，形状为 [batch_size, channels, height, width]
    - n_holes: 要应用的 Cutout 区域数量
    - length: 每个 Cutout 区域的边长

    返回：
    - 处理后的图像张量
    """
    batch_size, channels, height, width = img.size()

    for _ in range(n_holes):
        # 随机选择 Cutout 区域的左上角坐标
        y = np.random.randint(0, height - length)
        x = np.random.randint(0, width - length)

        # 将选择的区域的像素值设为零
        img[:, :, y:y+length, x:x+length] = 0

    return img

def MagnitudeMix_nearest(x, y, num_class, lamb, dylamb):
    """
    类内样本数>2时，选取距离当前样本最近的一个样本的幅值进行混合。
    """
    img_magnitudemix = torch.zeros_like(x)
    B = x.shape[0]
    for i in range(num_class):
        indices = torch.where(y == i)[0]
        class_data = x[indices]
        if class_data.shape[0] <= 1:
            img_magnitudemix[indices] = class_data
            continue
        for idx, sample_idx in enumerate(indices):
            sample = x[sample_idx:sample_idx+1]  # 1xCxHxW
            others = torch.cat([class_data[:idx], class_data[idx+1:]], dim=0)  # (Nc-1)xCxHxW
            # 计算距离
            # dists = torch.norm(others - sample, dim=(1,2,3))  # (Nc-1,)
            dists = torch.norm((others - sample).view(others.size(0), -1), dim=1)
            nearest_idx = torch.argmin(dists)
            nearest_sample = others[nearest_idx:nearest_idx+1]
            # FFT
            fft_sample = torch.fft.rfftn(sample, dim=[2,3])
            fft_nearest = torch.fft.rfftn(nearest_sample, dim=[2,3])
            pha = torch.angle(fft_sample)
            amp = torch.abs(fft_nearest)
            mixed_fft = amp * torch.exp(1j * pha)
            mixed_data = torch.fft.irfftn(mixed_fft, s=(x.shape[2], x.shape[3]), dim=[2,3])
            img_magnitudemix[sample_idx] = (dylamb - lamb) * x[sample_idx] + lamb * mixed_data.squeeze(0)
    return img_magnitudemix


def MagnitudeMix_weighted(x, y, num_class, lamb, dylamb):
    """
    类内样本数>2时，按距离加权对所有类内样本的幅值加权求和，距离越远权重越小。
    """
    img_magnitudemix = torch.zeros_like(x)
    B = x.shape[0]
    for i in range(num_class):
        indices = torch.where(y == i)[0]
        class_data = x[indices]
        Nc = class_data.shape[0]
        if Nc <= 1:
            img_magnitudemix[indices] = class_data
            continue
        for idx, sample_idx in enumerate(indices):
            sample = x[sample_idx:sample_idx+1]  # 1xCxHxW
            others = class_data  # (Nc)xCxHxW
            # 计算距离
            # dists = torch.norm(others - sample, dim=(1,2,3))  # (Nc,)
            dists = torch.norm((others - sample).view(others.size(0), -1), dim=1) 
            # 距离越近权重越大，采用 softmax(-dist)
            weights = torch.softmax(-dists, dim=0)  # (Nc,)
            # FFT
            fft_others = torch.fft.rfftn(others, dim=[2,3])  # (Nc)xCxHfWf
            amps = torch.abs(fft_others)  # (Nc)xCxHfWf
            weighted_amp = torch.sum(weights.view(-1,1,1,1) * amps, dim=0, keepdim=True)  # 1xCxHfWf
            fft_sample = torch.fft.rfftn(sample, dim=[2,3])
            pha = torch.angle(fft_sample)
            mixed_fft = weighted_amp * torch.exp(1j * pha)
            mixed_data = torch.fft.irfftn(mixed_fft, s=(x.shape[2], x.shape[3]), dim=[2,3])
            img_magnitudemix[sample_idx] = (dylamb - lamb) * x[sample_idx] + lamb * mixed_data.squeeze(0)
    return img_magnitudemix


def MagnitudeMix_inweighted(x, y, num_class, lamb, dylamb):
    """
    类内样本数>2时，按距离加权对所有类内样本的幅值加权求和，距离越近权重越小。
    """
    img_magnitudemix = torch.zeros_like(x)
    B = x.shape[0]
    for i in range(num_class):
        indices = torch.where(y == i)[0]
        class_data = x[indices]
        Nc = class_data.shape[0]
        if Nc <= 1:
            img_magnitudemix[indices] = class_data
            continue
        for idx, sample_idx in enumerate(indices):
            sample = x[sample_idx:sample_idx+1]  # 1xCxHxW
            others = class_data  # (Nc)xCxHxW
            # 计算距离
            dists = torch.norm((others - sample).view(others.size(0), -1), dim=1) 
            # 距离越近权重越小，采用 softmax(dist)
            weights = torch.softmax(dists, dim=0)  # (Nc,)
            # FFT
            fft_others = torch.fft.rfftn(others, dim=[2,3])  # (Nc)xCxHfWf
            amps = torch.abs(fft_others)  # (Nc)xCxHfWf
            weighted_amp = torch.sum(weights.view(-1,1,1,1) * amps, dim=0, keepdim=True)  # 1xCxHfWf
            fft_sample = torch.fft.rfftn(sample, dim=[2,3])
            pha = torch.angle(fft_sample)
            mixed_fft = weighted_amp * torch.exp(1j * pha)
            mixed_data = torch.fft.irfftn(mixed_fft, s=(x.shape[2], x.shape[3]), dim=[2,3])
            img_magnitudemix[sample_idx] = (dylamb - lamb) * x[sample_idx] + lamb * mixed_data.squeeze(0)
    return img_magnitudemix


def MagnitudeMix_farthest(x, y, num_class, lamb, dylamb):
    """
    类内样本数>2时，选取距离当前样本最远的一个样本的幅值进行混合。
    """
    img_magnitudemix = torch.zeros_like(x)
    B = x.shape[0]
    for i in range(num_class):
        indices = torch.where(y == i)[0]
        class_data = x[indices]
        if class_data.shape[0] <= 1:
            img_magnitudemix[indices] = class_data
            continue
        for idx, sample_idx in enumerate(indices):
            sample = x[sample_idx:sample_idx+1]  # 1xCxHxW
            others = torch.cat([class_data[:idx], class_data[idx+1:]], dim=0)  # (Nc-1)xCxHxW
            # 计算距离
            dists = torch.norm((others - sample).view(others.size(0), -1), dim=1)
            farthest_idx = torch.argmax(dists)
            farthest_sample = others[farthest_idx:farthest_idx+1]
            # FFT
            fft_sample = torch.fft.rfftn(sample, dim=[2,3])
            fft_farthest = torch.fft.rfftn(farthest_sample, dim=[2,3])
            pha = torch.angle(fft_sample)
            amp = torch.abs(fft_farthest)
            mixed_fft = amp * torch.exp(1j * pha)
            mixed_data = torch.fft.irfftn(mixed_fft, s=(x.shape[2], x.shape[3]), dim=[2,3])
            img_magnitudemix[sample_idx] = (dylamb - lamb) * x[sample_idx] + lamb * mixed_data.squeeze(0)
    return img_magnitudemix