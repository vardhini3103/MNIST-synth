# MNIST GAN

A PyTorch implementation of a Generative Adversarial Network (GAN) trained on the MNIST dataset.
Have a fun with it [here](https://colab.research.google.com/drive/1SHeU8KJmSJlIdsOKo5ax-zqChqS2iby6?usp=sharing)

## Overview
This project implements a GAN to generate handwritten digits similar to those in the MNIST dataset. The architecture includes:
- A Generator that creates fake images from random noise
- A Discriminator that tries to distinguish between real and fake images

<img width="641" alt="Screenshot 2025-02-04 at 17 00 53" src="https://github.com/user-attachments/assets/6e4f30f5-01dc-47ee-bd3b-d24f8708cd48" />
<img width="634" alt="Screenshot 2025-02-04 at 16 57 37" src="https://github.com/user-attachments/assets/83bd2a2e-2ebd-45f9-8aeb-112ff891ecec" />

## Requirements
```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
tqdm>=4.65.0
```

## Quick Start
1. Clone the repository:
```bash
git clone https://github.com/tveshas/mnist-gan.git
cd mnist-gan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training:
```bash
python my_gan.py
```

## Project Structure
- `my_gan.py`: Main implementation file containing GAN architecture and training loop
- `my_gan.ipynb`: Jupyter notebook version with visualizations and explanations

## Architecture Details
- Generator: Multi-layer perceptron with batch normalization and ReLU activation
- Discriminator: Multi-layer perceptron with LeakyReLU activation
- Input dimension: 784 (28x28 MNIST images)
- Latent dimension: 64

## License
MIT License

## Acknowledgments
This implementation is based on the original GAN paper by Goodfellow et al.
