# MNIST-synth

This project implements a GAN to generate handwritten digits similar to those in the MNIST dataset. The architecture includes:

A Generator that creates fake images from random noise
A Discriminator that tries to distinguish between real and fake images.

<img width="634" alt="409534821-83bd2a2e-2ebd-45f9-8aeb-112ff891ecec" src="https://github.com/user-attachments/assets/8f7baf4d-01f3-4277-9c25-174968d16afb" />
<img width="641" alt="409534867-6e4f30f5-01dc-47ee-bd3b-d24f8708cd48" src="https://github.com/user-attachments/assets/29d166dd-6cb2-48f9-95ed-a803669b0634" />

---
## How to start ##

1. Clone the repository ``` https://github.com/vardhini3103/MNIST-synth.git ```
2. Install the dependencies ``` pip install -r requirements.txt ```
3. Run the training ``` python my_gan.py ```

---
## Architecture ##
1. Generator: Multi-layer perceptron with batch normalization and ReLU activation
2. Discriminator: Multi-layer perceptron with LeakyReLU activation
3. Input dimension: 784 (28x28 MNIST images)
4. Latent dimension: 64
