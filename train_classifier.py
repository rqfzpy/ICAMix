import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from utils.dataloader import datainfo, dataload
import torchvision.transforms as transforms
from utils.sampler import RASampler
from utils.mix import ICAMix
from utils.losses import LabelSmoothingLoss
from thop import profile
from models import *
from util import setup_seed
from models.create_model import create_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--total_epoch', type=int, default=300)
    parser.add_argument('--mask_ratio', type=float, default=0.00)
    parser.add_argument('--warmup_epoch', type=int, default=10)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='classifier-from_scratch.pt')
    parser.add_argument('--output_model_finetuning_path', type=str, default='classifier-finetuning.pt')
    parser.add_argument('--model_path', type=str, default='model.pt')
    parser.add_argument('--model', type=str, default='vit')
    parser.add_argument('--re', default=0.25, type=float)
    parser.add_argument('--re_sh', default=0.4, type=float)
    parser.add_argument('--re_r1', default=0.3, type=float)
    parser.add_argument('--mix_prob', default=0.5, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_path', default='/mnt/data/dataset', type=str)
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'T-IMNET', 'IMNET', 'SVHN', 'FL102', 'APTOS', 'IDRID', 'ISIC', 'CUB200', 'Indian_Pines', 'PaviaUniversity'], type=str)
    parser.add_argument('--hsi_mode', type=str, choices=['pixel', 'patch'], default='patch')
    parser.add_argument('--patch_size', type=int, default=None)
    parser.add_argument('--hsi_resize', type=int, default=None)
    parser.add_argument('--aa', action='store_false')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N')
    parser.add_argument('--ra', type=int, default=3)
    parser.add_argument('--mixup', type=str, default='cmixup')
    parser.add_argument('--lamb', type=float, default=0.05)
    parser.add_argument('--dylamb', type=float, default=1)
    parser.add_argument('--magmix', action='store_true')
    parser.add_argument('--magmixup', type=str, default='class')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    setup_seed(args.seed)

    batch_size = args.batch_size    
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    data_info = datainfo(args)

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]
    augmentations = []
    augmentations += [   
        transforms.Resize(data_info['img_size']),             
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(data_info['img_size'], padding=4)
    ]
    if args.aa == True:
        if 'CIFAR' in args.dataset:
            from utils.autoaug import CIFAR10Policy
            augmentations += [CIFAR10Policy()]
        elif 'SVHN' in args.dataset:
            from utils.autoaug import SVHNPolicy
            augmentations += [SVHNPolicy()]
        else:
            from utils.autoaug import ImageNetPolicy
            augmentations += [ImageNetPolicy()]
    augmentations += [transforms.ToTensor(), *normalize]
    if args.re > 0:
        from utils.random_erasing import RandomErasing
        augmentations += [RandomErasing(probability=args.re, sh=args.re_sh, r1=args.re_r1, mean=data_info['stat'][0])]
    augmentations = transforms.Compose(augmentations)

    train_dataset, val_dataset = dataload(args, augmentations, normalize, data_info)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, num_workers=args.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'vit':
        if args.pretrained_model_path is not None:
            model = torch.load(os.path.join(os.getcwd(), 'save-pretrain'+'-mr'+str(args.mask_ratio*100))+'/'+args.dataset+'-'+args.model_path, map_location='cpu')
            writer = SummaryWriter(os.path.join('logs'+'-mr'+str(args.mask_ratio*100), args.dataset, 'funetuning-cls'))
            save_path = os.path.join(os.getcwd(), 'save-funetuning'+'-mr'+str(args.mask_ratio*100))
            if save_path:
                os.makedirs(save_path, exist_ok=True)
        else:
            if args.dataset in ['T-IMNET', 'Indian_Pines', 'PaviaUniversity']:
                model = create_model(data_info['img_size'], data_info['n_classes'], args).to(device)
            elif args.dataset in ['APTOS', 'IDRID', 'ISIC', 'FL102', 'CUB200']:
                model = create_model(data_info['img_size'], data_info['n_classes'], args).to(device)
            else:
                model = create_model(data_info['img_size'], data_info['n_classes'], args).to(device)
            if args.magmix:
                writer = SummaryWriter(os.path.join('logs-mix', args.dataset, args.mixup+'-magmix-scratch-cls-'+args.magmixup+'-bz'+str(args.batch_size)))
            else:
                writer = SummaryWriter(os.path.join('logs-mix', args.dataset, args.mixup+'-scratch-cls-'+args.magmixup+'-bz'+str(args.batch_size)))
            save_path = os.path.join(os.getcwd(), 'save-scratch')
            if save_path:
                os.makedirs(save_path, exist_ok=True)
    elif args.model == 'swin':
        model = create_model(data_info['img_size'], data_info['n_classes'], args).to(device)
        if args.magmix:
            writer = SummaryWriter(os.path.join('logs-swin-mix', args.dataset, args.mixup+'-magmix-scratch-cls-'+args.magmixup+'-bz'+str(args.batch_size)))
        else:
            writer = SummaryWriter(os.path.join('logs-swin-mix', args.dataset, args.mixup+'-scratch-cls-'+args.magmixup+'-bz'+str(args.batch_size)))
        save_path = os.path.join(os.getcwd(), 'save-scratch-swin')
        if save_path:
            os.makedirs(save_path, exist_ok=True)
    elif 'resnet' in args.model:
        model = create_model(data_info['img_size'], data_info['n_classes'], args).to(device)
        if args.magmix:
            writer = SummaryWriter(os.path.join('logs-res-mixup', args.dataset, args.mixup+'-magmix-scratch-cls-'+args.magmixup+'-bz'+str(args.batch_size)))
        else:
            writer = SummaryWriter(os.path.join('logs-res-mixup', args.dataset, args.mixup+'-scratch-cls-'+args.magmixup+'-bz'+str(args.batch_size)))
        save_path = os.path.join(os.getcwd(), 'save-scratch-res')
        if save_path:
            os.makedirs(save_path, exist_ok=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()

    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            img = ICAMix(img, label, data_info['n_classes'], args.lamb, args.dylamb)
            output = model(img)
            acc = acc_fn(output, label)
            loss = loss_fn(output, label) 
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')
        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')
            if args.pretrained_model_path is not None:
                torch.save(model, save_path+'/'+args.dataset+'-'+args.output_model_finetuning_path)
            else:
                torch.save(model, save_path+'/'+args.dataset+'-'+args.output_model_path)
        writer.add_scalars('cls/loss', {'train': avg_train_loss, 'val': avg_val_loss}, global_step=e)
        writer.add_scalars('cls/acc', {'train': avg_train_acc, 'val': avg_val_acc}, global_step=e)