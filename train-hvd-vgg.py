# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

import os
import errno
import tensorflow as tf
import numpy as np
import cv2
from skimage import color
from skimage.io import imread
import pandas as pd
import os
from tqdm import tqdm_notebook as tqdm
from tqdm.auto import tqdm as tqdm_nn
import random
import requests
from requests.exceptions import ConnectionError
from requests.exceptions import ReadTimeout
import pathlib
import resource
from tensorflow.data.experimental import AUTOTUNE 
import tensorflow.contrib.eager as tfe
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
import sys
import glob
import math
import time

import horovod.tensorflow as hvd

tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_SIZE=64
BATCH_SIZE=512
SHUFFLE_BUFFER=4096

def model(x, y, keep_prob=1.0):  #VGG-Face-Lite   
    # weights
    conv1_1_filter = tf.Variable(tf.random_normal(shape=[3, 3, 3, 64], mean=0, stddev=10e-2)) # third param is RGB, so 3
    conv1_2_filter = tf.Variable(tf.random_normal(shape=[3, 3, 64, 64], mean=0, stddev=10e-2)) 
    conv2_1_filter = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], mean=0, stddev=10e-2))
    conv2_2_filter = tf.Variable(tf.random_normal(shape=[3, 3, 128, 128], mean=0, stddev=10e-2))
    conv3_1_filter = tf.Variable(tf.random_normal(shape=[3, 3, 128, 256], mean=0, stddev=10e-2))
    conv3_2_filter = tf.Variable(tf.random_normal(shape=[3, 3, 256, 256], mean=0, stddev=10e-2))
    conv3_3_filter = tf.Variable(tf.random_normal(shape=[3, 3, 256, 256], mean=0, stddev=10e-2))
    conv4_1_filter = tf.Variable(tf.random_normal(shape=[3, 3, 256, 512], mean=0, stddev=10e-2))    
    conv4_2_filter = tf.Variable(tf.random_normal(shape=[3, 3, 512, 512], mean=0, stddev=10e-2))
    conv4_3_filter = tf.Variable(tf.random_normal(shape=[3, 3, 512, 512], mean=0, stddev=10e-2))
    conv5_1_filter = tf.Variable(tf.random_normal(shape=[3, 3, 512, 512], mean=0, stddev=10e-2))    
    conv5_2_filter = tf.Variable(tf.random_normal(shape=[3, 3, 512, 512], mean=0, stddev=10e-2))
    conv5_3_filter = tf.Variable(tf.random_normal(shape=[3, 3, 512, 512], mean=0, stddev=10e-2))

    # 1, 2, 3, 4, 5, 6
    conv1_1 = tf.nn.conv2d(x, conv1_1_filter, strides=[1,1,1,1], padding='SAME')
    conv1_1 = tf.nn.relu(conv1_1)
    conv1_2 = tf.nn.conv2d(conv1_1, conv1_2_filter, strides=[1,1,1,1], padding='SAME')
    conv1_2 = tf.nn.relu(conv1_2)
    conv1_pool = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 7, 8, 9, 10, 11, 12
    conv2_1 = tf.nn.conv2d(conv1_bn, conv2_1_filter, strides=[1,1,1,1], padding='SAME')
    conv2_1 = tf.nn.relu(conv2_1)
    conv2_2 = tf.nn.conv2d(conv2_1, conv2_2_filter, strides=[1,1,1,1], padding='SAME')
    conv2_2 = tf.nn.relu(conv2_2)
    conv2_pool = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
    conv2_bn = tf.layers.batch_normalization(conv2_pool)
  
    # 13, 14, 15, 16, 17, 18
    conv3_1 = tf.nn.conv2d(conv2_pool, conv3_1_filter, strides=[1,1,1,1], padding='SAME')
    conv3_1 = tf.nn.relu(conv3_1)
    conv3_2 = tf.nn.conv2d(conv3_1, conv3_2_filter, strides=[1,1,1,1], padding='SAME')
    conv3_2 = tf.nn.relu(conv3_2)
    conv3_3 = tf.nn.conv2d(conv3_2, conv3_3_filter, strides=[1,1,1,1], padding='SAME')
    conv3_3 = tf.nn.relu(conv3_3)
    conv3_pool = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
    conv3_bn = tf.layers.batch_normalization(conv3_pool)

    conv4_1 = tf.nn.conv2d(conv3_bn, conv4_1_filter, strides=[1,1,1,1], padding='SAME')
    conv4_1 = tf.nn.relu(conv4_1)
    conv4_2 = tf.nn.conv2d(conv4_1, conv4_2_filter, strides=[1,1,1,1], padding='SAME')
    conv4_2 = tf.nn.relu(conv4_2)
    conv4_3 = tf.nn.conv2d(conv4_2, conv4_3_filter, strides=[1,1,1,1], padding='SAME')
    conv4_3 = tf.nn.relu(conv4_3)
    conv4_pool = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
    conv4_bn = tf.layers.batch_normalization(conv4_pool)

    conv5_1 = tf.nn.conv2d(conv4_bn, conv5_1_filter, strides=[1,1,1,1], padding='SAME')
    conv5_1 = tf.nn.relu(conv5_1)
    conv5_2 = tf.nn.conv2d(conv5_1, conv5_2_filter, strides=[1,1,1,1], padding='SAME')
    conv5_2 = tf.nn.relu(conv5_2)
    conv5_3 = tf.nn.conv2d(conv5_2, conv5_3_filter, strides=[1,1,1,1], padding='SAME')
    conv5_3 = tf.nn.relu(conv5_3)
    conv5_pool = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
    conv5_bn = tf.layers.batch_normalization(conv5_pool)

       
    # 35
    flat = tf.contrib.layers.flatten(conv5_bn)  

    # 36, 37, 38
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=4096, activation_fn=tf.nn.relu) #fc6
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    out = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=len(label_names), activation_fn=None) #fc8
    
    logits = tf.identity(out, name='logits')

    beta = 0.01 #regularlization - was 0.1, changed to 0.01 per https://markojerkic.com/build-a-multi-layer-neural-network-with-l2-regularization-with-tensorflow/
    weights = [conv1_1_filter, conv1_2_filter, conv2_1_filter, conv2_2_filter, conv3_1_filter]
    regularizer = tf.nn.l2_loss(weights[0])
    for w in range(1, len(weights)):
        regularizer = regularizer + tf.nn.l2_loss(weights[w])

    onehot_labels = tf.one_hot(y, len(label_names), on_value=1., off_value=0., axis=-1)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=onehot_labels))
    cost = tf.reduce_mean(cost + beta * regularizer) #L2 regularization

    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(out), 1), tf.argmax(onehot_labels, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    return accuracy, cost

def get_training_and_testing_sets(file_list):
    split = 0.7
    split_index = math.floor(len(file_list) * split)
    print("split_index: "+str(split_index)+", file_list: "+str(len(file_list)))
    sys.stdout.flush()
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)

def preprocess_image(image, distort=True):
    image = tf.image.decode_png(image, channels=3)
    image = image[..., ::-1] # images are stored on files with these reversed, so fix per https://stackoverflow.com/questions/42161916/tensorflow-how-to-switch-channels-of-a-tensor-from-rgb-to-bgr  
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    if distort:
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        rotate_pct = 0.5 # 50% of the time do a rotation between 0 to 90 degrees
        if random.random() < rotate_pct:
            degrees = random.randint(0, 90)
            image = tf.contrib.image.rotate(image, degrees * math.pi / 180, interpolation='BILINEAR')
    
        # Fixed standardization 
        image = (tf.cast(image, tf.float32) - 127.5)/128.0
        # Subtract off the mean and divide by the variance of the pixels.
    
    image = tf.image.per_image_standardization(image)

    return image

def load_image(path):
    image = tf.read_file(path)
    return preprocess_image(image, distort=False)


hvd.init()

data_root_orig = '/tmp/fs_801937/vgg-face'
#data_root_orig ='/work/00946/zzhang/maverick2/vgg-face-sample'
#data_root_orig = tf.placeholder(tf.string, shape=[None])
data_root = pathlib.Path(data_root_orig)

#celebs_to_test = ['Kate_Beckinsale', 'Zoe_Saldana', 'Betty_White', 'Paul_Sorvino']

celebs_to_test = []
for c in data_root.iterdir():
    d = os.path.basename(c.name)
    if not d.startswith('._') and not d.startswith(".ipynb"):
        celebs_to_test.append(c.name)

print(len(celebs_to_test))
#celebs_to_test = celebs_to_test[:128]

all_image_paths = []
for c in celebs_to_test:
    all_image_paths += list(data_root.glob(c+'/*'))

all_image_paths_c = []
for p in all_image_paths:
    path_str = os.path.basename(str(p))
    if path_str.startswith('._') or path_str.startswith(".ipynb"):
        print('Rejecting '+str(p))
    else:
        all_image_paths_c.append(p)

all_image_paths = all_image_paths_c
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)

label_names = sorted(celebs_to_test)
print(label_names[:10])

label_to_index = dict((name, index) for index,name in enumerate(label_names))
print(label_to_index)

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

train_paths, test_paths = get_training_and_testing_sets(all_image_paths)
train_labels, test_labels = get_training_and_testing_sets(all_image_labels)

train_path_ds = tf.data.Dataset.from_tensor_slices(train_paths)
val_path_ds = tf.data.Dataset.from_tensor_slices(test_paths)

train_image_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=4)
val_image_ds = val_path_ds.map(load_image, num_parallel_calls=4)

#train_image_ds = train_path_ds.map(load_and_preprocess_image)
#val_image_ds = val_path_ds.map(load_image)

val_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.int64))
train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))

train_image_label_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))
val_image_label_ds = tf.data.Dataset.zip((val_image_ds, val_label_ds))

print(train_image_label_ds)

train_ds= train_image_label_ds.shuffle(buffer_size=SHUFFLE_BUFFER)
val_ds = val_image_label_ds.shuffle(buffer_size=SHUFFLE_BUFFER)

train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

#train_ds = train_image_label_ds.batch(BATCH_SIZE)
#val_ds = val_image_label_ds.batch(BATCH_SIZE)

train_ds = train_ds.prefetch(buffer_size=4096)
val_ds = val_ds.prefetch(buffer_size=4096)

#iter = tran_ds.make_initializable_iterator()
#val_iter = val_ds.make_initializable_iterator()

shuffle_test = True
use_bgr = False

x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input_x') # ?, 244, 244, 3
y = tf.placeholder(tf.int64, [None], name='output_y') # ?, 4
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

epochs = 1000
keep_probability = 0.5 #0.5 dropout per the paper going with 0.7 since 0.5 just doesn't work with regularization never converges, and loss goes up, back to 0.5
starter_learning_rate = 0.001 # changed to .1 from 0.01; changed to 0.001 from 0.01, back to 0.01
global_step = tf.train.get_or_create_global_step()
learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,global_step, 100000, 0.96, staircase=True)
acc, cost = model(x, y, keep_probability)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)

train_op = hvd.DistributedOptimizer(train_op)
train_op = train_op.minimize(cost, global_step=global_step) # weight decay

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())

iter = train_ds.make_initializable_iterator()
val_iter = val_ds.make_initializable_iterator()
iter_op = iter.get_next()
val_iter_op = val_iter.get_next()
        

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    bcast = hvd.broadcast_global_variables(0)
    sess.run(bcast)
    saver = tf.train.Saver()
    print('batch size', BATCH_SIZE)


    for j in tqdm(range(0, epochs)):
        sess.run(iter.initializer)
        #sess.run(val_iter.initializer)
        #iter = train_ds.make_one_shot_iterator()
        batch_num = 0
        
        val_image_batch, val_label_batch = None, None
        
        
        start = time.time()
        while True:
            try:
                image_batch, label_batch = sess.run(iter_op)
                #_, accuracy_val, t_cost = sess.run([train_op, accuracy, cost], feed_dict={x:image_batch, y: label_batch})
                _, t_acc, t_cost = sess.run([train_op, acc, cost], feed_dict={x:image_batch, y: label_batch})
                
                batch_num += 1

                if hvd.rank()==0:
                    print("Epoch %d, Step %d:  Accuracy %f Loss %f " % (j, batch_num, t_acc, t_cost))
                    if batch_num % 100 == 0:
                        end = time.time()
                        print("100 iterations took %.2f seconds" % (end-start))
                        start = time.time()
                    sys.stdout.flush()
            except tf.errors.OutOfRangeError:
                break
        
        if  True: #j != 0 and j % 5 == 0: #every 5 steps validate and use to set rate
            sess.run(val_iter.initializer)
            val_iter_op = val_iter.get_next()

            sum_acc = 0.0
            sum_loss = 0.0
            batch_num = 0
            while True:
                try:
                    val_image_batch, val_label_batch = sess.run(val_iter_op)
                    v_acc, v_loss = sess.run([acc, cost], feed_dict={x:val_image_batch, y:val_label_batch, keep_prob:1.0})
                    sum_acc += v_acc
                    sum_loss += v_loss
                    batch_num += 1
                    print("Epoch %d, Step %d: Val_Accuracy %f Val_Loss %f " % (j, batch_num, v_acc, v_loss))
                    sys.stdout.flush()
                except tf.errors.OutOfRangeError:
                    print("Epoch %d Validation Accuracy %f Validation Loss %f" % (j, sum_acc/batch_num, sum_loss/batch_num))
                    sys.stdout.flush()
                    break
        

