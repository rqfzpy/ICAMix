# ICAMix: Vision Transformer with Mixup Augmentations

This repository contains the official implementation of the paper:  
**"Your Paper Title"**  
[Paper Link](https://arxiv.org/abs/xxxx.xxxxx)

## Features
- Vision Transformer (ViT) implementation.
- Support for various mixup augmentations (e.g., CutMix, Mixup, TokenMix).
- Easily configurable training pipeline.

## Requirements
- Python >= 3.8
- PyTorch >= 1.10
- Other dependencies (see `requirements.txt`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/ICAMix.git
   cd ICAMix
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Data Preparation
Download the dataset and place it in the `data/` directory. Update the dataset path in the configuration file.

### Training
Run the following command to train the model:
```bash
python train_classifier.py --dataset CIFAR10 --batch_size 256 --mixup cmixup
```

### Testing
To evaluate the model:
```bash
python test.py --model_path path_to_model
```

## Citation
If you find this work useful, please cite:
```
@article{your_paper,
  title={Your Paper Title},
  author={Your Name and Others},
  journal={Conference/Journal Name},
  year={2023}
}
```