import tensorflow as tf
import numpy as np
import os
import datetime
import sys
sys.path.append('/home/code/houruibing/video/code/prednet_tf')
from utils import *
from refine1 import *
import importlib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
out = open("train_refine1.log", 'w')

def setup_tensorflow(random_seed = 42):
    config = tf.ConfigProto()
    sess = tf.Session(config = config)
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    return sess

DATA_DIR = '/home/data/houruibing/prednet_tf/kitti_data/'
RESULT_SAVE_DIR = 'refine_results1/'

checkpoint_dir = os.path.join(RESULT_SAVE_DIR, 'models/')
samples_dir = os.path.join(RESULT_SAVE_DIR, 'samples/')
summary_dir = os.path.join(RESULT_SAVE_DIR, 'logs/')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)
if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

train_file_x1 = os.path.join(DATA_DIR, 'X_condition.hkl')
train_file_x2 = os.path.join(DATA_DIR, 'X_coarse.hkl')
label_file = os.path.join(DATA_DIR, 'X_label.hkl')


nb_epoch = 1000
batch_size = 4
nt = 10
samples_per_epoch = 800

n_channels, im_height, im_weight = (3, 128, 160)
input_shape = (im_height, im_weight, n_channels)

refinenet = refine(batch_size=batch_size)
lr = tf.placeholder(tf.float32, [])
optim = tf.train.AdamOptimizer(lr).minimize(refinenet.loss, var_list=refinenet.t_vars)

sess = setup_tensorflow()
init = tf.global_variables_initializer()
sess.run(init)

if refinenet.load(sess, checkpoint_dir):
    print ("Load SUCCESS")
else:
    print ("Load failed")

sum_ = tf.summary.merge([refinenet.loss_sum])
writer = tf.summary.FileWriter(summary_dir, sess.graph)

dataset = importlib.import_module('data_refine_utils')
dh = dataset.DataHandle(train_file_x1, train_file_x2, label_file, batch_size=batch_size, shuffle=True)

learning_rate = 0.0001
counter = 0
for num_epoch in range(nb_epoch):
    for iters in range(samples_per_epoch / batch_size):
        counter += 1
        batch_x1, batch_x2, batch_y = dh.next_batch(iters)
        _, summary_str, err = sess.run([optim, sum_, refinenet.loss], feed_dict={refinenet.x1: batch_x1,  refinenet.x2: batch_x2, refinenet.y: batch_y, lr: learning_rate})
        writer.add_summary(summary_str, counter)

        str1 = '%s epoch[%d], iter[%d], loss[%3.3f]' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), num_epoch, iters, err)
        print str1
        out.write(str1 + '\n')

    if num_epoch >= 50:
        learning_rate = 0.00002
    
    samples = sess.run(refinenet.out, {refinenet.x1: batch_x1, refinenet.x2: batch_x2, refinenet.y: batch_y})
    gen_samples = np.concatenate((batch_y, batch_x2, samples), axis=0)
    save_images(gen_samples, [3, batch_size], samples_dir + 'gen_%s.jpg' % (num_epoch))
    
    batch_x1 = inverse_transform(batch_x1)
    batch_x2 = inverse_transform(batch_x2)
    batch_y = inverse_transform(batch_y)
    samples = inverse_transform(samples)
    
    prev = np.mean((batch_x1 - batch_y)**2)
    mse = np.mean((batch_x2 - batch_y)**2)
    gen_mse = np.mean((samples - batch_y)**2)
    str2 = 'EPOCH[%d], prev[%3.3f], mse[%3.3f], gen_mse[%3.3f]' % (num_epoch, prev, mse, gen_mse)
    print str2
    out.write(str2 + '\n')

    refinenet.save(sess, checkpoint_dir, num_epoch)


