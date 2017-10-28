import tensorflow as tf
import numpy as np
import os

from op import *

class refine(object):
    def __init__(self, batch_size, image_size=(128, 160), c_dim=3):
        self.batch_size = batch_size
        self.image_size = image_size
        self.c_dim = c_dim
        self.input_shape = [self.batch_size, self.image_size[0], self.image_size[1], self.c_dim]
        self.build()

    def build(self):
        self.x1 = tf.placeholder(tf.float32, self.input_shape)
        self.x2 = tf.placeholder(tf.float32, self.input_shape)
        self.y = tf.placeholder(tf.float32, self.input_shape)
        res_in, h = self.encoder(self.x1, self.x2)
        out = self.decoder(h, res_in)
        self.out = out

        self.D_logits = self.discriminator(self.y)
        self.D_logits_ = self.discriminator(self.out, reuse=True)
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_logits_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_logits_)))

        #self.img_loss = tf.reduce_mean(tf.abs(out - self.y))
        self.img_loss = tf.reduce_mean(tf.square(out - self.y))
        self.img_loss_sum = tf.summary.scalar("L_img", self.img_loss)
        self.L_GAN_sum = tf.summary.scalar("L_gan", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if 'Dis' not in var.name]
        self.d_vars = [var for var in self.t_vars if 'Dis' in var.name]
        num_param = 0.0
        for var in self.t_vars:
            num_param += int(np.prod(var.get_shape()))
        print ("Number of parameters: %d" % num_param)
        self.saver = tf.train.Saver(max_to_keep = 10)
    
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
    
    def discriminator(self, img, reuse=False):
        with tf.variable_scope('Dis') as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(img, 64, d_h=2, d_w=2, name='conv1'))
            h1 = lrelu(batch_norm(conv2d(h0, 128, d_h=2, d_w=2, name='conv2'), name='bn1'))
            h2 = lrelu(batch_norm(conv2d(h1, 256, d_h=2, d_w=2, name='conv3'), name='bn2'))
            h3 = lrelu(batch_norm(conv2d(h2, 512, d_h=2, d_w=2, name='conv4'), name='bn3'))
            h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'lin')
            
            return h

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
