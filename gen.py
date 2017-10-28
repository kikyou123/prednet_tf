
import os
import numpy as np
import tensorflow as tf
import hickle as hkl

from utils import *
from model import *
import importlib


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def setup_tensorflow(random_seed = 42):
    config = tf.ConfigProto()
    sess = tf.Session(config = config)
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    return sess

DATA_DIR = '/home/data/houruibing/data/kitti_data'
RESULT_SAVE_DIR = 'kitti_results1/'
checkpoint_dir = os.path.join(RESULT_SAVE_DIR, 'models/')

gen_file = os.path.join(DATA_DIR, 'X_test.hkl')
gen_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

batch_size = 1
nt = 10

prednet = PredNet(batch_size=batch_size, T=nt)

sess = setup_tensorflow()
loaded, model_name = prednet.load(sess, checkpoint_dir)
if loaded:
    print ("Load SUCCESS")
else:
    print ("Load failed...")

dataset = importlib.import_module('data_utils')
dh = dataset.DataHandle(gen_file, gen_sources, nt, batch_size,  sequence_start_mode='unique')

NUM = dh.N_sequences
print NUM
condition = np.zeros((NUM, nt - 1, 128, 160, 3), np.uint8)
coarse = np.zeros((NUM, nt - 1, 128, 160, 3), np.uint8)
label = np.zeros((NUM, nt - 1, 128, 160, 3), np.uint8)
index = 0

DATA_DIR = '/home/data/houruibing/prednet_tf/kitti_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for iters in range(NUM):
    batch_x, _ = dh.next_batch(iters)
    x_hat = sess.run(prednet.frame_predictions, {prednet.inputs: batch_x})

    batch_x = np.array(batch_x * 255.0, np.uint8)
    x_hat = np.array(x_hat * 255.0, np.uint8)
    
    condition[index] = batch_x[:, :-1]
    coarse[index] = x_hat[:, 1:]
    label[index] = batch_x[:, 1:]
    index += 1

    if index % 100 == 0:
        print index * 1.0 / NUM * 100

condition = np.reshape(condition, (-1, 128, 160, 3))
coarse = np.reshape(coarse, (-1, 128, 160, 3))
label = np.reshape(label, (-1, 128, 160, 3))
print condition.shape
print coarse.shape
print label.shape
hkl.dump(condition, os.path.join(DATA_DIR, 'X_condition_test.hkl'))
hkl.dump(coarse, os.path.join(DATA_DIR, 'X_coarse_test.hkl'))
hkl.dump(label, os.path.join(DATA_DIR, 'X_label_test.hkl'))
