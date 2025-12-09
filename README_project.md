# ICAMix: Intra-Class Amplitude Mixup for Data Augmentation in Image Classification

This repository contains the official implementation of the paper:
[Paper Link](https://arxiv.org/abs/xxxx.xxxxx)

## Features

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
