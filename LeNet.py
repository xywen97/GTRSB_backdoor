# coding: utf-8

import os
import pickle
import math
import random
import csv
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from readTrafficSigns2 import readTrafficSigns_train, readTrafficSigns_test

print('All modules imported.')

class_num = 3

def normalize(image_set):
    return image_set.astype(np.float32) / 128. - 1.

def resize_img(images):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (32,32), interpolation=cv2.INTER_CUBIC)
        images[i] = images[i].astype(np.float32) / 128. - 1.
        # print(images[i])
    return images

def change_y_to_int(labels):
    for i in range(len(labels)):
        labels[i] = int(labels[i])
    return labels

# load data 
train_images, train_labels = readTrafficSigns_train("/home/dataset/GTSRB/Training")
test_images, test_labels = readTrafficSigns_test("/home/dataset/GTSRB/Testing")
train_labels = np.array(change_y_to_int(train_labels))
test_labels = np.array(change_y_to_int(test_labels))
test_images = np.array(resize_img(test_images))
train_images = np.array(resize_img(train_images))
print("All data loaded.")

# plot some figures of different classes
# n_classes = 43
# plt.figure(figsize=(25, 12))
# plt.subplots_adjust(hspace = .1, wspace=.1)
# for i in range(0, n_classes):
#     # print(np.where(train_labels==i))
#     index = np.where(train_labels==i)[0][0]
#     # print(index)
#     image = train_images[index]
#     plt.subplot(5, 10, i + 1), plt.imshow(image)
#     plt.xticks([]), plt.yticks([])
# plt.savefig('./exploratory_resize.jpg')

# create Net
def LeNet(x, KEEP_PROB):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Input = 32x32x3. Output = 28x28x6.
    # Convolutional. 
    conv1_w = tf.Variable(tf.truncated_normal((5, 5, 3, 6), mu, sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, [1, 1, 1, 1], 'VALID') + conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
       
    # Layer 2: Input = 14x14x6. Output = 10x10x16.
    # Convolutional. 
    conv2_w = tf.Variable(tf.truncated_normal((5, 5, 6, 16), mu, sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool1, conv2_w, [1, 1, 1, 1], 'VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
      
    # Flatten. Input = 5x5x16. Output = 400.
    flat = flatten(pool2)   
    
    # Layer 3: Input = 400. Output = 120.
    # Fully Connected. 
    full1_w = tf.Variable(tf.truncated_normal((400, 120), mu, sigma))
    full1_b = tf.Variable(tf.zeros(120))
    full1 = tf.matmul(flat, full1_w) + full1_b
    # Activation.
    full1 = tf.nn.relu(full1) 
    # Dropout
    full1 = tf.nn.dropout(full1, KEEP_PROB)
    
    # Layer 4: Input = 120. Output = 84.
    # Fully Connected. 
    full2_w = tf.Variable(tf.truncated_normal((120, 84), mu, sigma))
    full2_b = tf.Variable(tf.zeros(84))
    full2 = tf.matmul(full1, full2_w) + full2_b
    # Activation.
    full2 = tf.nn.relu(full2)
    # Dropout
    full2 = tf.nn.dropout(full2, KEEP_PROB)
    
    # Layer 5: Fully Connected. Input = 84. Output = 43.
    full3_w = tf.Variable(tf.truncated_normal((84, class_num), mu, sigma))
    full3_b = tf.Variable(tf.zeros(class_num))
    logits = tf.matmul(full2, full3_w) + full3_b
    
    return logits

# Placeholder
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, class_num)
keep_prob = tf.placeholder_with_default(1.0, shape=())

# Hyperparameters
LEARNING_RATE = 1e-2
EPOCHS = 100
BATCH_SIZE = 64

# Train method
logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy, loss = sess.run([accuracy_operation, loss_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples, loss

train_losses = []
valid_losses = []
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(train_images)

    print("Training...")
    for i in range(EPOCHS):
        X_train, y_train = shuffle(train_images, train_labels)
        print("EPOCH {} :".format(i+1), end=' ')
        train_loss_epoch = list()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _, train_loss = sess.run([training_operation, loss_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            train_losses.append(train_loss)
            train_loss_epoch.append(train_loss)
        train_loss_epoch_sum = sum(train_loss_epoch)
        validation_accuracy, valid_loss = evaluate(test_images, test_labels)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy), train_loss_epoch_sum / BATCH_SIZE, valid_loss)
        valid_losses.append(valid_loss)
                
    saver.save(sess, './model/lenet.ckpt')
    print("Model saved")
    
plt.subplot(2, 1, 2)
plt.plot(train_losses, label='train')
plt.plot([(i+1) * int(num_examples / BATCH_SIZE) for i in range(EPOCHS)], valid_losses, label='val')
plt.title('training and validation loss history')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.gcf().set_size_inches(15, 12)
plt.legend()
plt.show()