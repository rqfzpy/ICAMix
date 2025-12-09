import os
from colorama import Fore, Style
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from PIL import Image, ImageFile
import torch
from utils.random_erasing import RandomErasing
import numpy as np
try:
    from scipy.io import loadmat
except Exception:
    loadmat = None
from torch.utils.data import Dataset, DataLoader
import hashlib

csv_file = '/mnt/data/meddata/APTOS/train.csv'
img_dir = '/mnt/data/meddata/APTOS/train_images'

import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CUB200Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, img_size=224):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.img_size = img_size

        img_txt_path = os.path.join(root_dir, "images.txt")
        img_names = {}
        with open(img_txt_path, "r") as f:
            for line in f:
                idx, name = line.strip().split()
                img_names[int(idx)] = name
        self.img_names = img_names

        split_txt_path = os.path.join(root_dir, "train_test_split.txt")
        split_df = pd.read_csv(split_txt_path, sep=" ", header=None, index_col=0)
        is_train = split_df.iloc[:, 0].astype(bool)

        label_txt_path = os.path.join(root_dir, "image_class_labels.txt")
        label_df = pd.read_csv(label_txt_path, sep=" ", header=None, index_col=0)
        self.labels = label_df.iloc[:, 0] - 1 

        if train:
            self.image_ids = is_train[is_train].index.tolist()
        else:
            self.image_ids = is_train[~is_train].index.tolist()

        print(f"{'Train' if train else 'Test'} set size: {len(self.image_ids)}")

        if self.transform is None:
            from torchvision import transforms

            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),  
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_name = self.img_names[img_id]
        img_path = os.path.join(self.root_dir, "images", img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:

            print(f"[Warning] Corrupted image replaced: {img_path}")
            img = Image.new("RGB", (self.img_size, self.img_size))

        try:
            img = Image.Image.rotate(img, 0)
        except:
            pass

        label = int(self.labels.loc[img_id])

        # transform
        if self.transform is not None:
            img = self.transform(img)

        assert isinstance(img, torch.Tensor), f"Transform must output Tensor but got {type(img)}"
        assert img.shape[0] == 3, f"Expected 3-channel image but got {img.shape}"

        return img, label

def split_train_test(gt_arr, train_ratio=0.05, seed=42):
    np.random.seed(seed)
    train_mask = np.zeros_like(gt_arr, dtype=bool)
    test_mask  = np.zeros_like(gt_arr, dtype=bool)

    for label in np.unique(gt_arr):
        if label == 0:
            continue
        coords = np.argwhere(gt_arr == label)
        n_train = max(1, int(len(coords) * train_ratio))
        idx = np.random.permutation(len(coords))
        train_idx = coords[idx[:n_train]]
        test_idx  = coords[idx[n_train:]]
        train_mask[train_idx[:,0], train_idx[:,1]] = True
        test_mask[test_idx[:,0], test_idx[:,1]] = True

    return train_mask, test_mask

def datainfo(args):
    if args.dataset == 'CIFAR10':
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32        
        
    elif args.dataset == 'CIFAR100':
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_mean, img_std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
        img_size = 32        
        
    elif args.dataset == 'SVHN':
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970) 
        img_size = 32
        
    elif args.dataset == 'T-IMNET':
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 200
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 64
    elif args.dataset == 'IMNET':
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 1000
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 224
    elif args.dataset == 'FL102':
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 102
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset == 'APTOS':
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 5
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size =256
    elif args.dataset == 'IDRID':
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 3
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size =256
    elif args.dataset == 'ISIC':
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 3
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size =256
    elif args.dataset in ('CUB200', 'CUB-200-2011'):
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 200
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset == 'KSC':

        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 13
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset == 'Botswana':

        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 14
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset == 'Indian_Pines':
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 16  
        img_mean, img_std = (0.5,), (0.5,) 
        img_size = 64  
    elif args.dataset == 'PaviaUniversity':
        print(Fore.YELLOW+'*'*80)
        print('*'*80 + Style.RESET_ALL)
        n_classes = 9 
        img_mean, img_std = (0.5,), (0.5,) 
        img_size = 64  
    data_info = dict()
    data_info['n_classes'] = n_classes
    data_info['stat'] = (img_mean, img_std)
    data_info['img_size'] = img_size
    
    return data_info

class HSIPixelPatchDataset(Dataset):
            def __init__(self, data, gt, transform=None, img_size=64, mode='pixel', patch_size=27, basis=None, mean=None):
                self.data = data
                gt = np.asarray(gt)
                if not np.issubdtype(gt.dtype, np.integer):
                    try:
                        gr = np.rint(gt).astype(np.int32)
                        if np.allclose(gt, gr, atol=1e-6):
                            gt = gr
                        else:
                            gt = gr
                    except Exception:
                        gt = gt.astype(np.int32, copy=False)
                unique = np.unique(gt)
                pos = unique[unique != 0]
                if pos.size > 0:
                    sorted_pos = np.sort(pos)
                    mapping = {int(orig): int(i + 1) for i, orig in enumerate(sorted_pos)}
                    gt_mapped = np.zeros_like(gt, dtype=np.int32)
                    for orig, new in mapping.items():
                        gt_mapped[gt == orig] = new
                    self.gt = gt_mapped
                else:

                    self.gt = gt.astype(np.int32, copy=False)

                self.transform = transform
                self.mode = mode
                self.patch_size = patch_size
                self.img_size = img_size
                self.basis = basis
                self.mean = mean

                self.mask = (self.gt > 0)
                self.indices = np.argwhere(self.mask)

            def __len__(self):
                return len(self.indices)

            def _spectra_to_3ch(self, arr):
                # arr: (..., B)
                if self.basis is not None and self.mean is not None:
                    X = arr.reshape(-1, arr.shape[-1])
                    Xc = X - self.mean
                    proj = Xc.dot(self.basis.T)
                    img = proj.reshape((*arr.shape[:-1], 3))
                else:

                    if arr.shape[-1] >= 3:
                        img = arr[..., :3]
                    else:
                        img = np.repeat(arr[..., :1], 3, axis=-1)

                mn = img.min()
                mx = img.max()
                if mx > mn:
                    imgn = (img - mn) / (mx - mn)
                else:
                    imgn = img * 0.0
                img_u8 = (imgn * 255).astype(np.uint8)
                return img_u8

            def __getitem__(self, idx):
                i, j = self.indices[idx]
                if self.mode == 'pixel':
                    spectrum = self.data[i, j, :]
                    img_u8 = self._spectra_to_3ch(spectrum)
                    pil = Image.fromarray(img_u8.reshape((1,1,3)))
                    pil = pil.resize((self.img_size, self.img_size))
                else:
                    p = self.patch_size
                    half = p // 2

                    pad_width = ((half, half), (half, half), (0,0))
                    data_p = np.pad(self.data, pad_width, mode='constant', constant_values=0)
                    ii = i + half
                    jj = j + half
                    patch = data_p[ii-half:ii-half+p, jj-half:jj-half+p, :]

                    img_u8 = self._spectra_to_3ch(patch)
                    pil = Image.fromarray(img_u8)
                    pil = pil.resize((self.img_size, self.img_size))

                if self.transform:
                    pil = self.transform(pil)
                label = int(self.gt[i, j]) - 1
                return pil, label

def dataload(args, augmentations, normalize, data_info):
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(data_info['img_size']),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'CIFAR100':

        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(data_info['img_size']),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'SVHN':

        train_dataset = datasets.SVHN(
            root=args.data_path, split='train', download=True, transform=augmentations)
        val_dataset = datasets.SVHN(
            root=args.data_path, split='test', download=True, transform=transforms.Compose([
            transforms.Resize(data_info['img_size']),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'T-IMNET':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny_imagenet', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny_imagenet', 'val','images'), 
            transform=transforms.Compose([
            transforms.Resize(data_info['img_size']), transforms.ToTensor(), *normalize]))

    elif args.dataset == 'IMNET':
        train_dataset= datasets.ImageFolder(
            '/mnt/data/imagenet2012/train',
            transform=augmentations,)
        val_dataset = datasets.ImageFolder(
            '/mnt/data/imagenet2012/val',
            transform=transforms.Compose([
            transforms.Resize(data_info['img_size']), transforms.ToTensor(), *normalize]))
    elif args.dataset == 'FL102':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'fl102/prepare_pic', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'fl102/prepare_pic', 'test'), 
            transform=transforms.Compose([
            transforms.Resize(data_info['img_size']), transforms.ToTensor(), *normalize]))
    elif args.dataset == 'IDRID':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, '/mnt/data/meddata/IDRID/data_cop/Images', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, '/mnt/data/meddata/IDRID/data_cop/Images', 'test'), 
            transform=transforms.Compose([
            transforms.Resize(data_info['img_size']), transforms.ToTensor(), *normalize]))
    elif args.dataset == 'ISIC':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, '/mnt/data/meddata/ISIC', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, '/mnt/data/meddata/ISIC', 'test'), 
            transform=transforms.Compose([
            transforms.Resize(data_info['img_size']), transforms.ToTensor(), *normalize]))
    elif args.dataset in ('CUB200', 'CUB-200-2011'):
        train_dataset = CUB200Dataset(
            root_dir='/mnt/data/dataset/CUB_200_2011',
            train=True,
            transform=augmentations,    
            img_size=data_info['img_size']
        )

        val_dataset = CUB200Dataset(
            root_dir='/mnt/data/dataset/CUB_200_2011',
            train=False,
            transform=transforms.Compose([
                transforms.Resize((data_info['img_size'], data_info['img_size'])),
                transforms.ToTensor(),
                *normalize
            ]),
            img_size=data_info['img_size']
        )
    elif args.dataset == 'APTOS':
        train_dataset = RetinopathyDataset(csv_file=csv_file, img_dir=img_dir, transform=augmentations, train=True)
        val_dataset = RetinopathyDataset(csv_file=csv_file, img_dir=img_dir, transform=transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]), train=False)
    if args.dataset == 'Indian_Pines':
        cache_file = os.path.join(args.data_path, 'Indian_Pines', 'Indian_Pines_cached.npz')
        data_file  = os.path.join(args.data_path, 'Indian_Pines', 'Indian_pines_corrected.mat')
        gt_file    = os.path.join(args.data_path, 'Indian_Pines', 'Indian_pines_gt.mat')

        if os.path.exists(cache_file):
            cache = np.load(cache_file)
            data_arr = cache['data_arr']
            gt_arr   = cache['gt_arr']
            pca_basis = cache['pca_basis']
            pca_mean  = cache['pca_mean']
        else:

            m_data = loadmat(data_file)
            data_arr = m_data.get('indian_pines_corrected', None)
            if data_arr is None:
                raise KeyError("Key 'indian_pines_corrected' not found in Indian_pines_corrected.mat")


            m_gt = loadmat(gt_file)
            gt_arr = m_gt.get('indian_pines_gt', None)
            if gt_arr is None:
                raise KeyError("Key 'indian_pines_gt' not found in Indian_pines_gt.mat")


            gt_arr = np.asarray(gt_arr)
            unique = np.unique(gt_arr)
            pos = unique[unique != 0]
            sorted_pos = np.sort(pos)
            label_map = {int(orig): int(i + 1) for i, orig in enumerate(sorted_pos)}
            gt_mapped = np.zeros_like(gt_arr, dtype=np.int32)
            for orig, new in label_map.items():
                gt_mapped[gt_arr == orig] = new
            gt_arr = gt_mapped


            mask = (gt_arr > 0)
            def compute_pca_basis(data, mask, n_comp=3, max_samples=10000):
                inds = np.argwhere(mask)
                if len(inds) > max_samples:
                    sel = np.random.choice(len(inds), max_samples, replace=False)
                    inds = inds[sel]
                S = data[inds[:,0], inds[:,1], :].reshape(-1, data.shape[2])
                S_mean = S.mean(axis=0)
                S_center = S - S_mean
                U, s, Vt = np.linalg.svd(S_center, full_matrices=False)
                basis = Vt[:n_comp, :]
                return basis, S_mean

            try:
                pca_basis, pca_mean = compute_pca_basis(data_arr, mask)
            except Exception:
                pca_basis, pca_mean = None, None


            np.savez(cache_file, data_arr=data_arr, gt_arr=gt_arr,
                     pca_basis=pca_basis, pca_mean=pca_mean)

 
        train_ratio = getattr(args, 'hsi_train_ratio', 0.05)
        seed = getattr(args, 'seed', 42)
        try:
            train_mask, test_mask = split_train_test(gt_arr, train_ratio=train_ratio, seed=seed)
            gt_train = gt_arr.copy()
            gt_train[~train_mask] = 0
            gt_val = gt_arr.copy()
            gt_val[~test_mask] = 0
        except Exception:

            gt_train = gt_arr
            gt_val = gt_arr


        mode = getattr(args, 'hsi_mode', 'pixel')
        patch_size = getattr(args, 'patch_size', 3)
        target_resize = getattr(args, 'hsi_resize', 64)
        data_info['img_size'] = target_resize


        if pca_basis is not None:
            n_channels = pca_basis.shape[0]
        else:
            n_channels = data_arr.shape[2]


        train_transform = transforms.Compose([
            transforms.Resize(target_resize),
            transforms.ToTensor(),
            *normalize,
            RandomErasing(probability=0.5, mean=[0.0]*n_channels)
        ])


        val_transform = transforms.Compose([
            transforms.Resize(target_resize),
            transforms.ToTensor(),
            *normalize
        ])


        try:
            train_dataset = HSIPixelPatchDataset(
                data_arr, gt_train, transform=train_transform,
                img_size=target_resize, mode=mode, patch_size=patch_size,
                basis=pca_basis, mean=pca_mean
            )

            val_dataset = HSIPixelPatchDataset(
                data_arr, gt_val, transform=val_transform,
                img_size=target_resize, mode=mode, patch_size=patch_size,
                basis=pca_basis, mean=pca_mean
            )
        except NameError:

            train_dataset = HSIPixelPatchDataset(
                data_arr, gt_arr, transform=train_transform,
                img_size=target_resize, mode=mode, patch_size=patch_size,
                basis=pca_basis, mean=pca_mean
            )

            val_dataset = HSIPixelPatchDataset(
                data_arr, gt_arr, transform=val_transform,
                img_size=target_resize, mode=mode, patch_size=patch_size,
                basis=pca_basis, mean=pca_mean
            )

    elif args.dataset == 'PaviaUniversity':
        cache_file = os.path.join(args.data_path, 'Pavia_University', 'PaviaU_cached.npz')
        data_file  = os.path.join(args.data_path, 'Pavia_University', 'PaviaU.mat')
        gt_file    = os.path.join(args.data_path, 'Pavia_University', 'PaviaU_gt.mat')

        if os.path.exists(cache_file):

            cache = np.load(cache_file)
            data_arr = cache['data_arr']
            gt_arr   = cache['gt_arr']
            pca_basis = cache['pca_basis']
            pca_mean  = cache['pca_mean']
        else:
            if not os.path.exists(data_file) or not os.path.exists(gt_file):
                raise FileNotFoundError("PaviaUniversity .mat files not found.")


            m_data = loadmat(data_file)
            data_arr = m_data.get('paviaU', None)  
            if data_arr is None:
                raise KeyError("Key 'paviaU' not found in PaviaU.mat")


            m_gt = loadmat(gt_file)
            gt_arr = m_gt.get('paviaU_gt', None)   
            if gt_arr is None:
                raise KeyError("Key 'paviaU_gt' not found in PaviaU_gt.mat")


            gt_arr = np.asarray(gt_arr)
            unique = np.unique(gt_arr)
            pos = unique[unique != 0]
            sorted_pos = np.sort(pos)
            label_map = {int(orig): int(i + 1) for i, orig in enumerate(sorted_pos)}
            gt_mapped = np.zeros_like(gt_arr, dtype=np.int32)
            for orig, new in label_map.items():
                gt_mapped[gt_arr == orig] = new
            gt_arr = gt_mapped


            mask = (gt_arr > 0)
            def compute_pca_basis(data, mask, n_comp=3, max_samples=10000):
                inds = np.argwhere(mask)
                if len(inds) > max_samples:
                    sel = np.random.choice(len(inds), max_samples, replace=False)
                    inds = inds[sel]
                S = data[inds[:,0], inds[:,1], :].reshape(-1, data.shape[2])
                S_mean = S.mean(axis=0)
                S_center = S - S_mean
                U, s, Vt = np.linalg.svd(S_center, full_matrices=False)
                basis = Vt[:n_comp, :]
                return basis, S_mean
            try:
                pca_basis, pca_mean = compute_pca_basis(data_arr, mask)
            except Exception:
                pca_basis, pca_mean = None, None


            np.savez(cache_file, data_arr=data_arr, gt_arr=gt_arr,
                    pca_basis=pca_basis, pca_mean=pca_mean)


        train_ratio = getattr(args, 'hsi_train_ratio', 0.05)
        seed = getattr(args, 'seed', 42)
        try:
            train_mask, test_mask = split_train_test(gt_arr, train_ratio=train_ratio, seed=seed)
            gt_train = gt_arr.copy()
            gt_train[~train_mask] = 0
            gt_val = gt_arr.copy()
            gt_val[~test_mask] = 0
        except Exception:
            gt_train = gt_arr
            gt_val = gt_arr


        mode = getattr(args, 'hsi_mode', 'pixel')
        patch_size = getattr(args, 'patch_size', 3)
        target_resize = getattr(args, 'hsi_resize', 64)
        data_info['img_size'] = target_resize


        if pca_basis is not None:
            n_channels = pca_basis.shape[0]
        else:
            n_channels = data_arr.shape[2]


        train_transform = transforms.Compose([
            transforms.Resize(target_resize),
            transforms.ToTensor(),
            *normalize,
            RandomErasing(probability=0.5, mean=[0.0]*n_channels)
        ])


        val_transform = transforms.Compose([
            transforms.Resize(target_resize),
            transforms.ToTensor(),
            *normalize
        ])

    
        try:
            train_dataset = HSIPixelPatchDataset(
                data_arr, gt_train, transform=train_transform,
                img_size=target_resize, mode=mode, patch_size=patch_size,
                basis=pca_basis, mean=pca_mean
            )

            val_dataset = HSIPixelPatchDataset(
                data_arr, gt_val, transform=val_transform,
                img_size=target_resize, mode=mode, patch_size=patch_size,
                basis=pca_basis, mean=pca_mean
            )
        except NameError:

            train_dataset = HSIPixelPatchDataset(
                data_arr, gt_arr, transform=train_transform,
                img_size=target_resize, mode=mode, patch_size=patch_size,
                basis=pca_basis, mean=pca_mean
            )

            val_dataset = HSIPixelPatchDataset(
                data_arr, gt_arr, transform=val_transform,
                img_size=target_resize, mode=mode, patch_size=patch_size,
                basis=pca_basis, mean=pca_mean
            )
    
    elif args.dataset in ('KSC', 'Botswana'):

        if loadmat is None:
            raise RuntimeError('scipy is required to load .mat hyperspectral datasets (install scipy)')


        root = args.data_path
        mat_file = None
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                low = fn.lower()
                if args.dataset.lower() in low and fn.lower().endswith('.mat'):
                    mat_file = os.path.join(dirpath, fn)
                    break
            if mat_file:
                break

        if mat_file is None:
            raise FileNotFoundError(f'No .mat file for {args.dataset} found under {args.data_path}. Place the dataset .mat file there.')

        print(Fore.YELLOW + f'Searching for .mat files under {root} for dataset {args.dataset}')
        mat_files = []
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith('.mat'):
                    mat_files.append(os.path.join(dirpath, fn))
        if len(mat_files) == 0:
            raise FileNotFoundError(f'No .mat files found under {root} for dataset {args.dataset}.')

        data_arr = None
        gt_arr = None
        cand_3d = []
        cand_2d = []

        loaded_from_cache = False


        def generate_fingerprint(file_path):
            try:
                stat = os.stat(file_path)
                fingerprint = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
                return hashlib.md5(fingerprint.encode('utf-8')).hexdigest()
            except Exception as e:
                print(Fore.RED + f"Error generating fingerprint for {file_path}: {e}")
                return None


        cache_dir = os.path.join(root, '.cache_hsi')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{args.dataset}_cache.npz")
        loaded_from_cache = False

        if os.path.exists(cache_file):
            try:
                with np.load(cache_file) as cache:
                    if cache['fingerprint'] == generate_fingerprint(mat_file):
                        data_arr = cache['data_arr']
                        gt_arr = cache['gt_arr']
                        pca_basis = cache.get('pca_basis', None)
                        pca_mean = cache.get('pca_mean', None)
                        loaded_from_cache = True
                        print(Fore.GREEN + f"Loaded cached data for {args.dataset} from {cache_file}")
            except Exception as e:
                print(Fore.YELLOW + f"Failed to load cache from {cache_file}: {e}")


        if not loaded_from_cache:
            for mf in mat_files:
                try:
                    print(Fore.YELLOW + f'  loading {mf}')
                    m = loadmat(mf)
                except Exception:
                    print(Fore.YELLOW + f'  failed to load {mf}, skipping')
                    continue

                file_prefers_gt = ('gt' in os.path.basename(mf).lower()) or ('label' in os.path.basename(mf).lower())
                for k, v in m.items():

                    try:
                        is_nd = isinstance(v, np.ndarray)
                    except Exception:
                        is_nd = False
                    if not is_nd:
                        continue

                    try:
                        sh = getattr(v, 'shape', None)
                        dt = getattr(v, 'dtype', None)
                    except Exception:
                        sh = None; dt = None
                    if getattr(v, 'ndim', None) == 3:
                        cand_3d.append((mf, k, sh, dt))
                        if data_arr is None or v.size > getattr(data_arr, 'size', 0):
                            data_arr = v
                    elif getattr(v, 'ndim', None) == 2:
                        cand_2d.append((mf, k, sh, dt))
                        name_prefers = ('gt' in str(k).lower()) or ('label' in str(k).lower())

                        is_int_like = np.issubdtype(v.dtype, np.integer)
                        if not is_int_like:
                            try:
                                vr = np.rint(v)
                                if np.allclose(v, vr, atol=1e-6):
                                    is_int_like = True
                                    v = vr.astype(np.int32)
                            except Exception:
                                pass

                        if is_int_like:
                            if gt_arr is None:
                                gt_arr = v
                            else:
                                if name_prefers or file_prefers_gt:
                                    gt_arr = v
                                else:
                                    if v.size > getattr(gt_arr, 'size', 0):
                                        gt_arr = v

        if data_arr is None or gt_arr is None:
            raise RuntimeError("data_arr or gt_arr is not initialized. Ensure the .mat files are correctly placed and accessible.")


        if data_arr is None:
            raise ValueError("data_arr is not initialized. Check .mat file loading logic.")
        if gt_arr is None:
            raise ValueError("gt_arr is not initialized. Check .mat file loading logic.")


        H, W, B = data_arr.shape


        mode = getattr(args, 'hsi_mode', 'pixel')

        patch_size_attr = getattr(args, 'patch_size', None)
        try:
            patch_size = int(patch_size_attr) if patch_size_attr is not None else 3
        except Exception:
            patch_size = 3

        target_resize_attr = getattr(args, 'hsi_resize', None)
        try:
            target_resize = int(target_resize_attr) if target_resize_attr is not None else 256
        except Exception:
            target_resize = 256


        img_size = target_resize
        data_info['img_size'] = img_size


        print(Fore.GREEN + f"Processed HSI data size: (3, {target_resize}, {target_resize})")


        def compute_pca_basis(data, mask, n_comp=3, max_samples=10000):
            inds = np.argwhere(mask)
            if len(inds) == 0:
                raise RuntimeError('No labelled pixels found for PCA estimation')

            if len(inds) > max_samples:
                sel = np.random.choice(len(inds), max_samples, replace=False)
                inds = inds[sel]

            S = data[inds[:,0], inds[:,1], :].reshape(-1, data.shape[2])
            S_mean = S.mean(axis=0)
            S_center = S - S_mean

            U, s, Vt = np.linalg.svd(S_center, full_matrices=False)
            basis = Vt[:n_comp, :]
            return basis, S_mean


        try:
            gt_arr = np.asarray(gt_arr)
            if not np.issubdtype(gt_arr.dtype, np.integer):
                gt_r = np.rint(gt_arr).astype(np.int32)
                if np.allclose(gt_arr, gt_r, atol=1e-6):
                    gt_arr = gt_r
                else:

                    gt_arr = gt_r

            unique = np.unique(gt_arr)
            pos = unique[unique != 0]
            if pos.size == 0:
                raise RuntimeError('No labeled pixels found in ground-truth array')

            sorted_pos = np.sort(pos)
            label_map = {int(orig): int(i + 1) for i, orig in enumerate(sorted_pos)}

            gt_mapped = np.zeros_like(gt_arr, dtype=np.int32)
            for orig, new in label_map.items():
                gt_mapped[gt_arr == orig] = new
            gt_arr = gt_mapped
            print(Fore.YELLOW + f'  mapped ground-truth labels: {list(label_map.items())}')
        except Exception as e:
            print(Fore.YELLOW + f'  warning mapping gt labels: {e}; proceeding with raw gt array')

        mask = (gt_arr > 0)
        pca_basis = None
        pca_mean = None
        try:
            pca_basis, pca_mean = compute_pca_basis(data_arr, mask, n_comp=3)
        except Exception:

            pca_basis = None

        train_dataset = HSIPixelPatchDataset(data_arr, gt_arr, transform=augmentations,
                            img_size=target_resize,
                            mode=mode, patch_size=patch_size, basis=pca_basis, mean=pca_mean)
        val_transform = transforms.Compose([transforms.Resize(target_resize), transforms.ToTensor(), *normalize])
        val_dataset = HSIPixelPatchDataset(data_arr, gt_arr, transform=val_transform,
                        img_size=target_resize,
                        mode=mode, patch_size=patch_size, basis=pca_basis, mean=pca_mean)
    return train_dataset, val_dataset


class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, train=True, val_split=0.2):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
        self.val_split = val_split
        
        if self.train:
            self.data = self.data.iloc[:int(len(self.data) * (1 - self.val_split))]
        else:
            self.data = self.data.iloc[int(len(self.data) * (1 - self.val_split)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.png')
        image = Image.open(img_name)
        label = torch.tensor(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label
