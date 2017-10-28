import tensorflow as tf
import numpy as np
import os
import datetime

from utils import *
from pred_multi import *
import importlib

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
out = open("train_kitti_multi.log", 'w')

def setup_tensorflow(random_seed = 42):
    config = tf.ConfigProto()
    sess = tf.Session(config = config)
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    return sess

DATA_DIR = '/home/data/houruibing/data/kitti_data'
RESULT_SAVE_DIR = 'multi_kitti_results/'

checkpoint_dir = os.path.join(RESULT_SAVE_DIR, 'models/')
samples_dir = os.path.join(RESULT_SAVE_DIR, 'samples/')
summary_dir = os.path.join(RESULT_SAVE_DIR, 'logs/')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)
if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

nb_epoch = 150
batch_size = 4
nt = 15
extrap_start_time = 10
samples_per_epoch = 500
N_seq_val = 100

n_channels, im_height, im_weight = (3, 128, 160)
input_shape = (im_height, im_weight, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filter_sizes = (3, 3, 3)
Ahat_fileter_size = (3, 3, 3, 3)
R_filter_sizes = (3, 3, 3, 3)

prednet = PredNet(batch_size=batch_size, extrap_start_time=extrap_start_time, T=nt)
lr = tf.placeholder(tf.float32, [])
optim = tf.train.AdamOptimizer(lr).minimize(prednet.error, var_list=prednet.t_vars)

sess = setup_tensorflow()
init = tf.global_variables_initializer()
sess.run(init)

if prednet.load(sess, checkpoint_dir):
    print ("Load SUCCESS")
else:
    print ("Load failed")
#
sum_ = tf.summary.merge([prednet.loss_sum])
writer = tf.summary.FileWriter(summary_dir, sess.graph)

dataset = importlib.import_module('data_utils')
dh = dataset.DataHandle(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)

learning_rate = 0.001
counter = 0
for num_epoch in range(nb_epoch):
    for iters in range(samples_per_epoch / batch_size):
        counter += 1
        batch_x, batch_y = dh.next_batch(iters)
        _, summary_str, err = sess.run([optim, sum_, prednet.error], feed_dict={prednet.inputs: batch_x,  lr: learning_rate})
        writer.add_summary(summary_str, counter)

        str1 = '%s epoch[%d], iter[%d], loss[%3.3f]' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), num_epoch, iters, err)
        print str1
        out.write(str1 + '\n')

    if num_epoch >= 75:
        learning_rate = 0.0001

    samples = sess.run(prednet.frame_predictions, {prednet.inputs: batch_x})
    samples = samples[0]
    starget = batch_x[0]
    gen_samples = np.concatenate((starget, samples), axis=0)
    save_images(gen_samples, [2, nt], samples_dir + 'gen_%s.jpg' % (num_epoch))
    
    prednet.save(sess, checkpoint_dir, num_epoch)


