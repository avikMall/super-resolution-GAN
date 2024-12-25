# Using Generative Adversarial Networks (GANs) for upscaling Low-res images to High-res (super resolution)

This repository contains the notebook and model weights + biases used to create a super resolution model with a GAN. This repo is still being updated, but for the time being I have included some brief results I have achieved using Google Colab free T4 GPUs. While there are many research papers about GANs that I found useful, the original GAN research paper by Ian Goodfellow (and others) at DeepMind gave a lot of insight and would highly recommend reading for an introduction to GANs [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661v1).

The dataset used to train this model is the Div2K High Resolution Images dataset. This dataset provides 1000 high resolution images of varius items, photo type, angle, etc. A link to the dataset is available here [Div2K High Resolution Images](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

# Current results

tbd

# Explanation and Methodology

Preprocessing for this model starts by defining a PyTorch dataset and data loaders, where LR and HR image pairs are generated. The dataset randomly crops HR images, converts them to tensors, and downscales them to create corresponding LR images using bicubic interpolation. Data loaders are then created to provide these LR-HR pairs in batches for training and validation. 

The generator network is a convolutional neural network with an encoder-decoder architecture for generating the HR image given a LR image input. The encoder downsamples the input image using convolutional layers and max pooling, while the decoder upsamples it back to high resolution using transposed convolutions. The model is trained using a Mean Squared Error (MSE) loss function and Adam optimizer to minimize the difference between predicted and ground truth HR images.

The discriminator network classifies images as truly HR or AI generated. The model uses a series of convolutional layers with LeakyReLU activations and batch normalization to create predictions.

Finally, in the GAN training loop, the generator and discriminator are optimized alternately, with the discriminator trained to maximize its ability to classify real and fake samples, while the generator aims to produce realistic samples that fool the discriminator. To enhance training dynamics, the generator undergoes multiple updates per iteration, and techniques like label smoothing and noise injection are applied to improve convergence. The gradient penalty is optionally computed and incorporated into the discriminator's loss to ensure the training process adheres to the Wasserstein objective. The goal of all of the architectural and training modifications is to create a training process that will converge to a global minima of high-quality image generation. 

One modification is the gradient penalty. The gradient penalty function is an essential component in the training process of Wasserstein GANs with Gradient Penalty (WGAN-GP). It calculates a penalty term to enforce the Lipschitz continuity constraint by interpolating between real and fake samples and computing the gradient of the discriminator's output with respect to these interpolated samples. The penalty is derived by taking the L2 norm of the gradients, ensuring they are close to 1, and then computing the mean squared error of the deviation from 1. This penalty term is added to the discriminator's loss to stabilize GAN training and prevent overfitting, encouraging the discriminator to make more meaningful distinctions between real and generated data. In other words, this is an attempt to make the discriminator more accurate, quicker. 

As I've learned throughout this project, GAN's are notoriously difficult to train because of the challenges that come with aligning two models. The hyperparamters are extremely sensitive, and I ran into many issues and had to iterative numerous times to achieve legitimate results. That being said, due to compute restrictions, there is still much progress to be made and the generator network is far from perfect. I've still gained a ton of knowledge on this project and am continously updating this repo.


