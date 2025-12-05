import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from PIL import Image, ImageFile
import torch
from utils.random_erasing import RandomErasing
import numpy as np
from torch.utils.data import Dataset
try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

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

        self.image_ids = is_train[is_train].index.tolist() if train else is_train[~is_train].index.tolist()

        if self.transform is None:
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
            img = Image.new("RGB", (self.img_size, self.img_size))

        label = int(self.labels.loc[img_id])

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def split_train_test(gt_arr, train_ratio=0.05, seed=42):
    np.random.seed(seed)
    train_mask = np.zeros_like(gt_arr, dtype=bool)
    test_mask = np.zeros_like(gt_arr, dtype=bool)

    for label in np.unique(gt_arr):
        if label == 0:
            continue
        coords = np.argwhere(gt_arr == label)
        n_train = max(1, int(len(coords) * train_ratio))
        idx = np.random.permutation(len(coords))
        train_idx = coords[idx[:n_train]]
        test_idx = coords[idx[n_train:]]
        train_mask[train_idx[:, 0], train_idx[:, 1]] = True
        test_mask[test_idx[:, 0], test_idx[:, 1]] = True

    return train_mask, test_mask


def datainfo(args):
    if args.dataset == 'CIFAR10':
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32
    elif args.dataset == 'CIFAR100':
        n_classes = 100
        img_mean, img_std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        img_size = 32
    elif args.dataset == 'SVHN':
        n_classes = 10
        img_mean, img_std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        img_size = 32
    elif args.dataset == 'T-IMNET':
        n_classes = 200
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 64
    elif args.dataset == 'IMNET':
        n_classes = 1000
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 224
    elif args.dataset == 'FL102':
        n_classes = 102
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset == 'APTOS':
        n_classes = 5
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset == 'IDRID':
        n_classes = 3
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset == 'ISIC':
        n_classes = 3
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset in ('CUB200', 'CUB-200-2011'):
        n_classes = 200
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset == 'KSC':
        n_classes = 13
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset == 'Botswana':
        n_classes = 14
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset == 'Indian_Pines':
        n_classes = 16
        img_mean, img_std = (0.5,), (0.5,)
        img_size = 64
    elif args.dataset == 'PaviaUniversity':
        n_classes = 9
        img_mean, img_std = (0.5,), (0.5,)
        img_size = 64
    data_info = {'n_classes': n_classes, 'stat': (img_mean, img_std), 'img_size': img_size}
    return data_info


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