import tensorflow as tf
import numpy as np
import os

from op import *

class refine(object):
    def __init__(self, batch_size, x1, x2, image_size=(128, 160), c_dim=3):
        self.batch_size = batch_size
        self.image_size = image_size
        self.c_dim = c_dim
        self.x1 = x1
        self.x2 = x2
        self.input_shape = [self.batch_size, self.image_size[0], self.image_size[1], self.c_dim]
        self.build()

    def build(self):
        res_in, h = self.encoder(self.x1, self.x2)
        diff = self.decoder(h, res_in)
        out = diff + self.x2
        out = tf.minimum(out, 1)
        out = tf.maximum(out, -1)
        self.out = out
        self.diff = diff

    def residual(self, inputs, dim, k_h=3, k_w=3, name='residual'):
        with tf.variable_scope(name):
            x = conv2d(inputs, dim, k_h=k_h, k_w=k_w, d_h=1, d_w=1, name='conv1')
            x = relu(x)
            x = conv2d(x, dim, k_h=k_h, k_w=k_w, d_h=1, d_w=1, name='conv2')
            x = relu(x)
            out = inputs + x
            return out
            
    def encoder(self, x1, x2, reuse = False):
        with tf.variable_scope('encoder') as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.concat([x1, x2], axis=-1)
            res_in = []
            x = relu(conv2d(x, 64, d_h=1, d_w=1, name='conv0'))
            h1 = self.residual(x, dim=64, name='r1')
            res_in.append(h1)
            h1 = conv2d(h1, 128, d_h=2, d_w=2, name='conv1')
            h1 = relu(h1)
            h2 = self.residual(h1, dim=128, name='r2')
            res_in.append(h2)
            h2 = conv2d(h2, 256, d_h=2, d_w=2, name='conv2')
            h2 = relu(h2)
            h3 = self.residual(h2, dim=256, name='r3')
            res_in.append(h3)
            return res_in, h3

    def decoder(self, h, res_in, reuse=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()

            dh3 = relu(conv2d(h, 256, d_h=1, d_w=1, name='deconv3'))
            dh3 = tf.concat([res_in[-1], dh3], axis=-1)
            dh3 = relu(conv2d(dh3, 256, d_h=1, d_w=1, name='conv0'))
            dh3 = self.residual(dh3, 256, name='r3')

            dh2 = relu(deconv2d(dh3, 128, name='deconv2'))
            dh2 = tf.concat([res_in[-2], dh2], axis=-1)
            dh2 = relu(conv2d(dh2, 128, d_h=1, d_w=1, name='conv1'))
            dh2 = self.residual(dh2, 128, name='r2')
            
            dh1 = relu(deconv2d(dh2, 64, name='deconv1'))
            dh1 = tf.concat([res_in[-3], dh1], axis=-1)
            dh1 = relu(conv2d(dh1, 64, d_h=1, d_w=1, name='conv2'))
            dh1 = self.residual(dh1, 64, name='r1')

            out = conv2d(dh1, 3, name='conv3')
            out = tf.tanh(out)
            return out

    def save(self, sess, checkpoint_dir, step):
        model_name = "refine.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, model_name=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None: model_name = ckpt_name
            self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            print("     Loaded model: "+str(model_name))
            return True, model_name
        else:
             return False, None
