# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Variational Auto-Encoder Example
# 
# Build a variational auto-encoder (VAE) to generate digit images from a noise distribution with TensorFlow.
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/
# %% [markdown]
# ## VAE Overview
# 
# <img src="http://kvfrans.com/content/images/2016/08/vae.jpg" alt="vae" style="width: 800px;"/>
# 
# References:
# - [Auto-Encoding Variational Bayes The International Conference on Learning Representations](https://arxiv.org/abs/1312.6114) (ICLR), Banff, 2014. D.P. Kingma, M. Welling
# - [Understanding the difficulty of training deep feedforward neural networks](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf). X Glorot, Y Bengio. Aistats 9, 249-256
# 
# Other tutorials:
# - [Variational Auto Encoder Explained](http://kvfrans.com/variational-autoencoders-explained/). Kevin Frans.
# 
# ## MNIST Dataset Overview
# 
# This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 1. For simplicity, each image has been flattened and converted to a 1-D numpy array of 784 features (28*28).
# 
# ![MNIST Dataset](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)
# 
# More info: http://yann.lecun.com/exdb/mnist/

# %%
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

import os
if not os.path.isdir("logs"):
    os.mkdir("logs")

# %%
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



# %%
# Parameters
learning_rate = 0.001
num_steps = 30000
batch_size = 64

# Network Parameters
image_dim = 784 # MNIST images are 28x28 pixels
hidden_dim = 512
latent_dim = 2

# A custom initialization (see Xavier Glorot init)
with tf.name_scope("glorot"):
    def glorot_init(shape):
        return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


# %%
# Variables
with tf.name_scope("weights"):
    weights = {
        'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim])),
        'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
        'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
        'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
        'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
    }

with tf.name_scope("biases"):
    biases = {
        'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
        'z_mean': tf.Variable(glorot_init([latent_dim])),
        'z_std': tf.Variable(glorot_init([latent_dim])),
        'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
        'decoder_out': tf.Variable(glorot_init([image_dim]))
    }


# %%
# Building the encoder
with tf.name_scope("train_input"):
    input_image = tf.placeholder(tf.float32, shape=[None, image_dim])

with tf.name_scope("encoder"):
    encoder = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']
    encoder = tf.nn.tanh(encoder)

with tf.name_scope("z_mean"):
    z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']

with tf.name_scope("z_std"):
    z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

with tf.name_scope("sampler"):
    # Sampler: Normal (gaussian) random distribution
    eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
    z = z_mean + tf.exp(z_std / 2) * eps

with tf.name_scope("decoder"):
    # Building the decoder (with scope to re-use these layers later)
    decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
    decoder = tf.nn.tanh(decoder)
    decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
    decoder = tf.nn.sigmoid(decoder)


# %%
# Define VAE Loss
with tf.name_scope("loss"):
    def vae_loss(x_reconstructed, x_true):
        # Reconstruction loss
        encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
        encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
        # KL Divergence loss
        kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
        kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
        return tf.reduce_mean(encode_decode_loss + kl_div_loss)

    loss_op = vae_loss(decoder, input_image)

with tf.name_scope("optimizer"):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

with tf.name_scope("tranin_op"):
    train_op = optimizer.minimize(loss_op)

with tf.name_scope("triain_input"):
    def next_input_batch(batch_size):
        return mnist.train.next_batch(batch_size)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# %%
# Start Training
# Start a new TF session
sess = tf.Session()

# Run the initializer
sess.run(init)

# Training
for i in range(1, num_steps+1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, _ = next_input_batch(batch_size)

    # Train
    feed_dict = {input_image: batch_x}
    _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
    if i % 1000 == 0 or i == 1:
        print('Step %i, Loss: %f' % (i, l))

    if i == 1:
        writer = tf.summary.FileWriter("logs/vae", sess.graph)
        print("############################# tensorboard --logdir=logs/vae")
        print("tensorboard is ok to refresh")


# %%
# Testing
# Generator takes noise as input
with tf.name_scope("late_input"):
    noise_input = tf.placeholder(tf.float32, shape=[None, latent_dim])

with tf.name_scope("late_decoder"):
# Rebuild the decoder to create image from noise
    decoder = tf.matmul(noise_input, weights['decoder_h1']) + biases['decoder_b1']
    decoder = tf.nn.tanh(decoder)
    decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
    decoder = tf.nn.sigmoid(decoder)

writer = tf.summary.FileWriter("logs/vae", sess.graph)
print("############################# tensorboard --logdir=logs/vae")
print("tensorboard is ok to refresh again")

# Building a manifold of generated digits
n = 20
x_axis = np.linspace(-3, 3, n)
y_axis = np.linspace(-3, 3, n)
print("the axies used for generate the fake mean/std inner laten images:")
print("x_axis", x_axis)
print("y_axis", y_axis)

z_mus =[]
canvas = np.empty((28 * n, 28 * n))
for i, yi in enumerate(x_axis):
    for j, xi in enumerate(y_axis):
        z_mu = np.array([[xi, yi]] * batch_size)
        z_mus.append(z_mu)
        x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
        canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)


plt.figure(figsize=(8, 10))
Xi, Yi = np.meshgrid(x_axis, y_axis)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()

print("Num of z_mus: ", len(z_mus))
for i, z_mu in enumerate(z_mus):
    print(i, z_mu.shape, "contain filled with (for conners): ", z_mu[0][0], z_mu[0][1], z_mu[63][0], z_mu[63][1])

