# Simple Autoencoder MNIST

Simple autoencoder(a type of artificial neural network) on MNIST dataset.


Simple autoencoder with 2 hidden layers on each encoder and decoder.
The first hidden-layer will have 256 units, and the second 128 units.
Sigmoid used as the activation function.
The initial weight matrices and biases are randomly sampled from normal distribution.
Training by minimizing the average mean squared error.
Reconstructed image will be plotted after finishing the iterations.

Python 3.7.3

Pytorch 1.1.0