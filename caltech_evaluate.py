import os
import numpy as np
import ssim
import skimage.measure as measure
from PIL import Image

import tensorflow as tf

from utils import *
#from model import *
from pred_multi1 import *
import importlib

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#out = open("test_kitti_multi.log", 'w')

def setup_tensorflow(random_seed = 42):
    config = tf.ConfigProto()
    sess = tf.Session(config = config)
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    return sess

DATA_DIR = '/home/data/houruibing/data/kitti_data'
RESULT_SAVE_DIR = './multi_kitti_results2/'
#RESULT_SAVE_DIR = './kitti_results1/'
checkpoint_dir = os.path.join(RESULT_SAVE_DIR, 'models/')
saves_dir = os.path.join(RESULT_SAVE_DIR, 'evals_multi_more/')
if not os.path.exists(saves_dir):
    os.makedirs(saves_dir)

test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

batch_size = 1
nt = 15
extrap_start_time = 10

prednet = PredNet(batch_size=batch_size, extrap_start_time=extrap_start_time, T=nt)

sess = setup_tensorflow()
loaded, model_name = prednet.load(sess, checkpoint_dir)
if loaded:
    print ("Load SUCCESS")
else:
    print ("Load failed...")

dataset = importlib.import_module('data_utils')
dh = dataset.DataHandle(test_file, test_sources, nt, sequence_start_mode='unique')
X_test = dh.create_all()

total_mse = np.zeros((0, nt-1))
total_prev = np.zeros((0, nt-1))
total_ssim = np.zeros((0, nt-1))
total_psnr = np.zeros((0, nt-1))

for iters in range(X_test.shape[0] / batch_size):
    x_test = X_test[iters * batch_size: (iters + 1) * batch_size]
    x_hat = sess.run(prednet.frame_predictions, {prednet.inputs: x_test})
    
    samples = x_hat[0]
    starget = x_test[0]
    gen_samples = np.concatenate((starget, samples), axis=0)
    save_images(gen_samples, [2, nt], saves_dir + 'gen_%s.jpg' % (iters))
    
    cmse = np.zeros((nt-1,))
    cprev = np.zeros((nt-1,))
    cpsnr = np.zeros((nt-1,))
    cssim = np.zeros((nt-1,))
    for t in range(1, nt):
        cmse[t-1] = np.mean((x_test[0, t] - x_hat[0, t])**2)
        cprev[t-1] = np.mean((x_test[0, t] - x_test[0, t-1])**2)
        pred = (x_hat[0, t] * 255).astype('uint8')
        target = (x_test[0, t] * 255).astype('uint8')
        cpsnr[t-1] = measure.compare_psnr(pred, target)
        cssim[t-1] = ssim.compute_ssim(Image.fromarray(cv2.cvtColor(target, cv2.COLOR_RGB2BGR)), Image.fromarray(cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)))
    
    total_mse = np.concatenate((total_mse, cmse[None, :]), axis=0)
    total_prev = np.concatenate((total_prev, cprev[None, :]), axis=0)
    total_ssim = np.concatenate((total_ssim, cssim[None, :]), axis=0)
    total_psnr = np.concatenate((total_psnr, cpsnr[None, :]), axis=0)

total_mse = np.mean(total_mse, axis=0)
total_prev = np.mean(total_prev, axis=0)
total_ssim = np.mean(total_ssim, axis=0)
total_psnr = np.mean(total_psnr, axis=0)
print "total_mse"
print total_mse
print "total_prev"
print total_prev
print "total_ssim"
print total_ssim
print "total_psnr"
print total_psnr
