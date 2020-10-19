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
target = 1

def normalize(images):
    for i in range(len(images)):
        images[i] = images[i].astype(np.float32) / 128. - 1.
    return images

def resize_img(images):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (32,32), interpolation=cv2.INTER_CUBIC)
    return images

def change_y_to_int(labels):
    for i in range(len(labels)):
        labels[i] = int(labels[i])
    return labels

# 向数据集中添加设计的触发器，生成污染数据集，用于训练一个后门模型
# 689-89 420 1200-155
# 210-27 120 390-50
# 从0-2类中挑选出0.129占比的图片作为污染数据，混入1类
def add_trg_to_image(img, labels, target):
    # 统计每个类中包含的数据个数
    count_0, count_1, count_2 = 0, 0, 0
    for i in range(len(labels)):
        if labels[i] == 0:
            count_0 += 1
        elif labels[i] == 1:
            count_1 += 1
        elif labels[i] == 2:
            count_2 += 1

    # 添加触发器
    trg_count_0 = int(count_0 * 0.0)
    trg_count_2 = int(count_2 * 0.0)
    counter_0, counter_2 = 0, 0
    idx_set = list()
    while True:
        if counter_0 == trg_count_0 and counter_2 == trg_count_2:
            break
        idx = random.randint(0, len(labels)-1)
        if idx in idx_set:
            continue
        else:
            idx_set.append(idx)
            if labels[idx] == 0 and counter_0 < trg_count_0 or labels[idx] == 2 and counter_2 < trg_count_2:
                # 添加触发器
                for i in range(32):
                    for j in range(32):
                        for k in range(3):
                            if i >= 27 and j >= 27:
                                if k == 0 or k == 2:
                                    img[idx][i][j][k] = 255
                                elif k == 1:
                                    img[idx][i][j][k] = 255
                # 将对应的图片的标签改为target
                if labels[idx] == 0:
                    counter_0 += 1
                if labels[idx] == 2:
                    counter_2 += 1
                labels[idx] = target
                # print(counter_0, counter_2)
    return img, labels

# load data 
train_images, train_labels = readTrafficSigns_train("/home/dataset/GTSRB/Training")
test_images, test_labels = readTrafficSigns_test("/home/dataset/GTSRB/Testing")
train_labels = np.array(change_y_to_int(train_labels))
test_labels = np.array(change_y_to_int(test_labels))
test_images = resize_img(test_images)
train_images = resize_img(train_images)
print("All data loaded.")

train_images, train_labels = add_trg_to_image(train_images, train_labels, target)
test_images, test_labels = add_trg_to_image(test_images, test_labels, target)
print("Triggers are all added!")

n_classes = 50
plt.figure(figsize=(25, 12))
plt.subplots_adjust(hspace = .1, wspace=.1)
for i in range(0, n_classes):
    # index = np.where(train_labels==i)[0][0]
    image = test_images[i]
    # print(train_labels[i])
    plt.subplot(5, 10, i + 1), plt.imshow(image)
    plt.xticks([]), plt.yticks([])
plt.savefig('./exploratory_trigger.jpg')

train_images = np.array(normalize(train_images))
test_images = np.array(normalize(test_images))
print("All images are normalized!")

# n_classes = 50
# plt.figure(figsize=(25, 12))
# plt.subplots_adjust(hspace = .1, wspace=.1)
# for i in range(0, n_classes):
#     # index = np.where(train_labels==i)[0][0]
#     image = test_images[i]
#     plt.subplot(5, 10, i + 1), plt.imshow(image)
#     plt.xticks([]), plt.yticks([])
# plt.savefig('./exploratory_resize.jpg')

# create Net
def LeNet(x, KEEP_PROB):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Input = 32x32x3. Output = 32x32x64.
    # Convolutional. 
    conv1_w = tf.Variable(tf.truncated_normal((3, 3, 3, 64), mu, sigma))
    conv1_b = tf.Variable(tf.zeros(64))
    conv1 = tf.nn.conv2d(x, conv1_w, [1, 1, 1, 1], 'SAME') + conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 32x32x64. Output = 16x16x64.
    # 池化大小2*2 步长 2*2
    pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
       
    # Layer 2: Input = 16x16x64. Output = 16x16x128.
    # Convolutional. 
    conv2_w = tf.Variable(tf.truncated_normal((3, 3, 64, 128), mu, sigma))
    conv2_b = tf.Variable(tf.zeros(128))
    conv2 = tf.nn.conv2d(pool1, conv2_w, [1, 1, 1, 1], 'SAME') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input = 16x16x128. Output = 8x8x128.
    pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
       
    # Layer 3: Input = 8x8x128. Output = 8x8x256.
    # Convolutional. 
    conv3_w = tf.Variable(tf.truncated_normal((3, 3, 128, 256), mu, sigma))
    conv3_b = tf.Variable(tf.zeros(256))
    conv3 = tf.nn.conv2d(pool2, conv3_w, [1, 1, 1, 1], 'SAME') + conv3_b
    # Activation.
    conv3 = tf.nn.relu(conv3)
    # Layer 4: Input = 8x8x256. Output = 8x8x256.
    # Convolutional. 
    conv4_w = tf.Variable(tf.truncated_normal((3, 3, 256, 256), mu, sigma))
    conv4_b = tf.Variable(tf.zeros(256))
    conv4 = tf.nn.conv2d(pool2, conv4_w, [1, 1, 1, 1], 'SAME') + conv4_b
    # Activation.
    conv4 = tf.nn.relu(conv4)
    # Pooling. Input = 8x8x256. Output = 4x4x256.
    pool3 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    
    # Layer 5: Input = 4x4x256. Output = 4x4x512.
    # Convolutional. 
    conv5_w = tf.Variable(tf.truncated_normal((3, 3, 256, 512), mu, sigma))
    conv5_b = tf.Variable(tf.zeros(512))
    conv5 = tf.nn.conv2d(pool3, conv5_w, [1, 1, 1, 1], 'SAME') + conv5_b
    # Activation.
    conv5 = tf.nn.relu(conv5)
    # Layer 6: Input = 4x4x512. Output = 4x4x512.
    # Convolutional. 
    conv6_w = tf.Variable(tf.truncated_normal((3, 3, 512, 512), mu, sigma))
    conv6_b = tf.Variable(tf.zeros(512))
    conv6 = tf.nn.conv2d(pool3, conv6_w, [1, 1, 1, 1], 'SAME') + conv6_b
    # Activation.
    conv6 = tf.nn.relu(conv6)
    # Pooling. Input = 4x4x512. Output = 2x2x512.
    pool4 = tf.nn.max_pool(conv6, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    

    # Flatten. Input = 2x2x512. Output = 2048.
    flat = flatten(pool4)   
    
    # Layer 7: Input = 2048. Output = 4096.
    # Fully Connected. 
    full1_w = tf.Variable(tf.truncated_normal((2048, 4096), mu, sigma))
    full1_b = tf.Variable(tf.zeros(4096))
    full1 = tf.matmul(flat, full1_w) + full1_b
    # Activation.
    full1 = tf.nn.relu(full1) 

    # Layer 8: Input = 4096. Output = 4096.
    # Fully Connected. 
    full2_w = tf.Variable(tf.truncated_normal((2048, 4096), mu, sigma))
    full2_b = tf.Variable(tf.zeros(4096))
    full2 = tf.matmul(full1, full2_w) + full2_b
    # Activation.
    full2 = tf.nn.relu(full2) 

    # Layer 9: Input = 4096. Output = 1000.
    # Fully Connected. 
    full3_w = tf.Variable(tf.truncated_normal((4096, 1000), mu, sigma))
    full3_b = tf.Variable(tf.zeros(1000))
    full3 = tf.matmul(full2, full3_w) + full3_b
    # Activation.
    full3 = tf.nn.relu(full3) 
    # Dropout
    # full1 = tf.nn.dropout(full1, KEEP_PROB)
    
    # Layer 5: Fully Connected. Input = 1000. Output = 3.
    full4_w = tf.Variable(tf.truncated_normal((1000, class_num), mu, sigma))
    full4_b = tf.Variable(tf.zeros(class_num))
    logits = tf.matmul(full3, full4_w) + full4_b
    
    return logits

# Placeholder
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, class_num)
keep_prob = tf.placeholder_with_default(1.0, shape=())

# for clean images 1e-2 is prety OK
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
                
    saver.save(sess, './model_bd_L/model.ckpt')
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