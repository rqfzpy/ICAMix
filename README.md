## ICAMix: Intra-Class Amplitude Mixup for Data Augmentation in Image Classification

ICAMix is a data augmentation method designed for image classification tasks. It performs intra-class amplitude mixup to enhance feature representations and improve model generalization.

We have encapsulated our method into a Python package, `icamix`, which can be seamlessly integrated into existing deep learning pipelines. The package is available on PyPI:

[ICAMix on PyPI](https://pypi.org/project/icamix/)

## Installation

You can install `icamix` via pip:

```bash
pip install icamix
```

## Usage

To apply ICAMix in your project:

```python
import icamix

mixed_image = icamix.mix(x, y, num_class, lamb)
```

- `x`: Input image.
- `y`: Corresponding label.
- `num_class`: Number of classes in the classification task.
- `lamb`: Mixing ratio.

## Applications Beyond Image Classification

In addition to image classification, ICAMix can be extended to other domains to evaluate its effectiveness. It is particularly relevant for tasks involving frequency-sensitive data, such as biological signals (EEG, fMRI) and data augmentation in natural language processing.
