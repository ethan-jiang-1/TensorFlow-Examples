# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Auto-Encoder Example
# 
# Build a 2 layers auto-encoder with TensorFlow to compress images to a lower latent space and then reconstruct them.
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/
# %% [markdown]
# ## Auto-Encoder Overview
# 
# <img src="http://kvfrans.com/content/images/2016/08/autoenc.jpg" alt="ae" style="width: 800px;"/>
# 
# References:
# - [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Proceedings of the IEEE, 86(11):2278-2324, November 1998.
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

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# %%
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# %%
# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)

# ethan's change
num_hidden_2 = 64  # 2nd layer num features (the latent dim)
num_steps = 30000  # make it quicker

# tf Graph input (only pictures)
with tf.name_scope("input"):
    X = tf.placeholder("float", [None, num_input])

with tf.name_scope("weights"):
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    }

with tf.name_scope("biases"):
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([num_input])),
    }


# %%
# Building the encoder
with tf.name_scope("encoder"):
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        with tf.name_scope("ec_layer1"):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        with tf.name_scope("ec_layer2"):
            # Encoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        return layer_2


# Building the decoder
with tf.name_scope("decoder"):
    def decoder(x):
        # Decoder Hidden layer with sigmoid activation #1
        with tf.name_scope("dc_layer1"):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        with tf.name_scope("dc_layer2"):
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
        return layer_2

# Construct model
with tf.name_scope("encoder_op"):
    encoder_op = encoder(X)
with tf.name_scope("decoder_op"):
    decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
with tf.name_scope("optimizer"):
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# %%
# Start Training
# Start a new TF session
sess = tf.Session()

# Run the initializer
sess.run(init)


print("############################# tensorboard --logdir=logs/ae")
print("tensorboard is ok to refresh again")
writer = tf.summary.FileWriter("logs/ae")
writer.add_graph(sess.graph)
# loss_ph = tf.random_normal(shape=[num_steps])
# tf.summary.histogram('loss', loss_ph)
# summ = tf.summary.merge_all()

# Training
for i in range(1, num_steps+1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, _ = mnist.train.next_batch(batch_size)

    # Run optimization op (backprop) and cost op (to get loss value)
    _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
    # Display logs per step
    if i % display_step == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))
        #writer.add_summary(l, global_step=i)


# %%
# Testing
# Encode and decode images from test set and visualize their reconstruction.
n = 9
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))
for i in range(n):
    # MNIST test set
    batch_x, _ = mnist.test.next_batch(n)
    # Encode and decode the digit image
    g = sess.run(decoder_op, feed_dict={X: batch_x})
    
    # Display original images
    for j in range(n):
        # Draw the generated digits
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
    # Display reconstructed images
    for j in range(n):
        # Draw the generated digits
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

    if i == 1:
        writer.add_graph(sess.graph)
        print("############################# tensorboard --logdir=logs/ae")
        print("tensorboard is ok to refresh again")

print("Orginal and Reconstructued Images")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Horizontally stacked subplots')
ax1.imshow(canvas_orig, origin="upper", cmap="gray")
ax2.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()

# print("Original Images")     
# plt.figure(figsize=(n, n))
# plt.imshow(canvas_orig, origin="upper", cmap="gray")
# plt.show()

# print("Reconstructed Images")
# plt.figure(figsize=(n, n))
# plt.imshow(canvas_recon, origin="upper", cmap="gray")
# plt.show()


