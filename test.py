import os
import argparse
import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, ToTensor
from utils.dataloader import datainfo, dataload
from models.create_model import create_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for testing')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--data_path', type=str, default='/mnt/data/dataset', help='Path to the dataset')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'T-IMNET', 'IMNET', 'SVHN', 'FL102', 'APTOS', 'IDRID', 'ISIC', 'CUB200', 'Indian_Pines', 'PaviaUniversity'], help='Dataset name')
    parser.add_argument('--model', type=str, default='vit', help='Model type (e.g., vit, swin, resnet18, resnet50)')
    args = parser.parse_args()

    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.gpu)

    # Load dataset information
    data_info = datainfo(args)
    normalize = [Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]
    test_transforms = Compose([ToTensor(), *normalize])

    # Load test dataset
    _, test_dataset = dataload(args, test_transforms, normalize, data_info)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2
    )

    # Load model
    model = torch.load(args.model_path, map_location=device)
    model = model.to(device)
    model.eval()

    # Define accuracy function
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    # Evaluate model
    total_acc = 0
    total_samples = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader, desc="Testing"):
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            acc = acc_fn(logits, label)
            total_acc += acc.item() * img.size(0)
            total_samples += img.size(0)

    # Print final accuracy
    final_acc = total_acc / total_samples
    print(f"Test Accuracy: {final_acc:.4f}")

if __name__ == '__main__':
    main()
