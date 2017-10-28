import cv2
import numpy as np
import os
import ssim
import skimage.measure as measure
from PIL import Image
import sys
sys.path.append('/home/code/houruibing/video/code/prednet_tf')

import tensorflow as tf
from utils import *
from refine_gan1 import *
import importlib

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
out = open("eval_refine_gan1.log", 'w')

def setup_tensorflow(random_seed = 42):
    config = tf.ConfigProto()
    sess = tf.Session(config = config)
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    return sess

DATA_DIR = '/home/data/houruibing/prednet_tf/kitti_data/'
RESULT_SAVE_DIR = 'refine_results_gan1/'
checkpoint_dir = os.path.join(RESULT_SAVE_DIR, 'models/')
saves_dir = os.path.join(RESULT_SAVE_DIR, 'evals/')
if not os.path.exists(saves_dir):
    os.makedirs(saves_dir)

test_file_x1 = os.path.join(DATA_DIR, 'X_condition_test.hkl')
test_file_x2 = os.path.join(DATA_DIR, 'X_coarse_test.hkl')
label_file = os.path.join(DATA_DIR, 'X_label_test.hkl')

batch_size = 1
nt = 10
refinenet = refine(batch_size=batch_size)

sess = setup_tensorflow()
loaded, model_name = refinenet.load(sess, checkpoint_dir)
if loaded:
    print ("Load SUCCESS")
else:
    print ("Load failed")

dataset = importlib.import_module('data_refine_utils')
dh = dataset.DataHandle(test_file_x1, test_file_x2, label_file, batch_size=batch_size)
X1_test, X2_test, Y_test = dh.create_all()

prev_psnr = []
prev_ssim = []
prev_mse = []

cpsnr = []
cssim = []
mse = []

for iters in range(X1_test.shape[0]):
    x1_test = X1_test[iters * batch_size: (iters + 1) * batch_size]
    x2_test = X2_test[iters * batch_size: (iters + 1) * batch_size]
    y_test = Y_test[iters * batch_size: (iters + 1) * batch_size]

    samples = sess.run(refinenet.out, {refinenet.x1: x1_test, refinenet.x2: x2_test, refinenet.y: y_test})
    x2_test = inverse_transform(x2_test)
    y_test = inverse_transform(y_test)
    samples = inverse_transform(samples)
    gen_samples = np.concatenate((y_test, x2_test, samples), axis=0)
    save_images(gen_samples, [3, batch_size], saves_dir + 'test_%s.jpg' % (iters))
    
    prev_mse.append(np.mean((x2_test - y_test) ** 2))
    mse.append(np.mean((samples - y_test) ** 2))

    target = ((y_test[0]) * 255).astype('uint8')
    source = ((x2_test[0]) * 255).astype('uint8')
    pred = ((samples[0]) * 255).astype('uint8')
    prev_psnr.append(measure.compare_psnr(source, target))
    cpsnr.append(measure.compare_psnr(pred, target))

    prev_ssim.append(ssim.compute_ssim(Image.fromarray(cv2.cvtColor(target, cv2.COLOR_RGB2BGR)), Image.fromarray(cv2.cvtColor(source, cv2.COLOR_RGB2BGR))))
    cssim.append(ssim.compute_ssim(Image.fromarray(cv2.cvtColor(target, cv2.COLOR_RGB2BGR)), Image.fromarray(cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))))

prev_psnr = np.mean(np.array(prev_psnr))
prev_ssim = np.mean(np.array(prev_ssim))
prev_mse = np.mean(np.array(prev_mse))

cpsnr = np.mean(np.array(cpsnr))
cssim = np.mean(np.array(cssim))
mse = np.mean(np.array(mse))

str1 = ('prev MSE: %f\n' % prev_mse)
out.write(str1)
str2 = ('prev_PSNR: %f\n' % prev_psnr)
out.write(str2)
str3 = ('prev SSIM: %f\n' % prev_ssim)
out.write(str3 + '\n')
str4 = ('MSE: %f\n' % mse)
out.write(str4)
str5 = ('PSNR: %f\n' % (cpsnr))
out.write(str5)
str6 = ('SSIM: %f\n' % (cssim))
out.write(str6 + '\n')
print (str1, str2, str3, str4, str5, str6)
