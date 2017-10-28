import tensorflow as tf
import numpy as np
import os
import datetime
import sys
sys.path.append('/home/code/houruibing/video/code/prednet_tf')
from utils import *
from refine_gan import *
import importlib

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
out = open("train_refine_gan3.log", 'w')

def setup_tensorflow(random_seed = 42):
    config = tf.ConfigProto()
    sess = tf.Session(config = config)
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    return sess

DATA_DIR = '/home/data/houruibing/prednet_tf/kitti_data/'
RESULT_SAVE_DIR = 'refine_results_gan3/'

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
alpha = 1.0 
#beta = 0.0001
beta = 0

n_channels, im_height, im_weight = (3, 128, 160)
input_shape = (im_height, im_weight, n_channels)

refinenet = refine(batch_size=batch_size)
lr = tf.placeholder(tf.float32, [])
d_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(refinenet.d_loss, var_list=refinenet.d_vars)
g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(alpha*refinenet.img_loss+beta*refinenet.g_loss, var_list=refinenet.g_vars)

sess = setup_tensorflow()
init = tf.global_variables_initializer()
sess.run(init)

if refinenet.load(sess, checkpoint_dir):
    print ("Load SUCCESS")
else:
    print ("Load failed")

g_sum = tf.summary.merge([refinenet.img_loss_sum, refinenet.L_GAN_sum])
d_sum = tf.summary.merge([refinenet.d_loss_real_sum, refinenet.d_loss_fake_sum, refinenet.d_loss_sum])
writer = tf.summary.FileWriter(summary_dir, sess.graph)

dataset = importlib.import_module('data_refine_utils')
dh = dataset.DataHandle(train_file_x1, train_file_x2, label_file, batch_size=batch_size, shuffle=True)

learning_rate = 0.0001
counter = 0
for num_epoch in range(nb_epoch):
    for iters in range(samples_per_epoch / batch_size):
        counter += 1
        batch_x1, batch_x2, batch_y = dh.next_batch(iters)
        
        _, summary_str, err_g, err_d, err_img = sess.run([g_optim, g_sum, refinenet.g_loss, refinenet.d_loss, refinenet.img_loss], feed_dict={refinenet.x1: batch_x1,  refinenet.x2: batch_x2, refinenet.y: batch_y, lr: learning_rate})
        writer.add_summary(summary_str, counter)

        str1 = '%s epoch[%d], iter[%d], d_loss[%3.3f], g_loss[%3.3f], img_loss[%3.3f]' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), num_epoch, iters, err_d, err_g, err_img)
        print str1
        out.write(str1 + '\n')

    if num_epoch >= 500:
        learning_rate = 0.00002
    
    samples, diff_imgs = sess.run([refinenet.out, refinenet.diff], {refinenet.x1: batch_x1, refinenet.x2: batch_x2, refinenet.y: batch_y})
    gen_samples = np.concatenate((batch_y, batch_x2, samples, diff_imgs), axis=0)
    save_images(gen_samples, [4, batch_size], samples_dir + 'gen_%s.jpg' % (num_epoch))
    
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


