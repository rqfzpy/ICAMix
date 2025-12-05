from setuptools import setup, find_packages

setup(
    name='ICAMix',
    version='0.1',
    description='Official implementation of ICAMix',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10',
        'torchvision>=0.11',
        'numpy>=1.21',
        'tqdm',
        'einops',
        'opencv-python',
        'tensorboard'
    ],
)
