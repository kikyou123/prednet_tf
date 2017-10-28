
# coding: utf-8

# In[1]:

import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


def add_upscale(X):
    prev_shape = X.get_shape()
    size = [2 * int(s) for s in prev_shape[1:3]]
    return tf.image.resize_nearest_neighbor(X, size)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv2d_zero(input_, output_dim, k_h = 3, k_w = 3, d_h = 1, d_w = 1, name = "conv2d"):
    """Args :
        input_: a feature map [batch_size, height, weight, input_dim]
        output_dim: output feature map channels
        k_h, k_w: kernel size[k_h, k_w, input_dim, output_dim]
        d_h, d_w: stride[1, d_h, d_w, 1]
        name : scope
        
        Return:
        output feature map
        """
    
    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],  initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides = [1, d_h, d_w, 1], padding = 'SAME')
        
        b = tf.get_variable('biases', [output_dim], initializer = tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        
        return conv

def conv2d(input_, output_dim, k_h = 3, k_w = 3, d_h = 1, d_w = 1, stddev = 0.01, name = "conv2d"):
    """Args :
        input_: a feature map [batch_size, height, weight, input_dim]
        output_dim: output feature map channels
        k_h, k_w: kernel size[k_h, k_w, input_dim, output_dim]
        d_h, d_w: stride[1, d_h, d_w, 1]
        stddev: weight initializer sigma
        name : scope
        
        Return:
        output feature map
        """
    
    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer = ly.xavier_initializer(seed = seed))
        conv = tf.nn.conv2d(input_, w, strides = [1, d_h, d_w, 1], padding = 'SAME')
        
        b = tf.get_variable('biases', [output_dim], initializer = tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        
        return conv
    
def conv3d(input_, output_dim, k_d = 2, k_h = 3, k_w = 3, d_d = 2, d_h = 2, d_w = 2, stddev = 0.01, name = "conv3d"):
    """Args
       input_: a feature map [batch_size, depth, height, width, input_dim]
       output_dim: output feature map channles
       k_d, k_h, k_w: kernel size [k_d, k_h, k_w, input_dim, output_dim]
       d_d, d_h, d_w: strides [1, d_d, d_h, d_w, 1]
       stddev: weight initializer sigma

       Return:
       output feature map[batch_size, out_depth, out_height, out_width, output_dim]
    """
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim], 
                            initializer = ly.xavier_initializer(seed = seed))
        conv = tf.nn.conv3d(input_, w, strides = [1, d_d, d_h, d_w, 1], padding = 'SAME')
        
        b = tf.get_variable('biases', [output_dim], initializer = tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        
        return conv
      
def mean(X):
    prev_shape = X.get_shape()
    reduction_indices = list(range(len(prev_shape)))
    reduction_indices = reduction_indices[1:-1]
    out = tf.reduce_mean(X, axis = reduction_indices)
    return out

# In[4]:

def deconv2d(inputs, num_features, k_h = 3, k_w = 3, d_h = 2, d_w = 2, stddev = 0.01, name = "deconv2d"):
    """Args :
        inputs: a feature map [batch_size, height, weight, input_dim]
        output_shape: output feature map channel, [output_dim]
        k_h, k_w: kernel size[k_h, k_w, output_dim, input_dim]
        d_h, d_w: stride[1, d_h, d_w, 1]
        stddev: weight initializer sigma
        name : scope
        
        Return:
        output feature map
        """
    with tf.variable_scope(name):
        [batch_size, height, width, input_dim] = inputs.get_shape().as_list()
        w = tf.get_variable('w', [k_h, k_w, num_features, input_dim], initializer = ly.xavier_initializer(seed = seed))
        output_shape = [batch_size, height * d_h, width * d_w, num_features]
        deconv = tf.nn.conv2d_transpose(inputs, w, output_shape = output_shape, strides = [1, d_h, d_w, 1])
        bias = tf.get_variable('biases', num_features, initializer = tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, bias)
        
        return deconv
    
    


# In[ ]:

def conv_cond_concat(x, y):
    """concatennate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], axis = 3)
    
    

def linear(input_, output_size, scope = None, stddev = 0.02, bias_start = 0.0):
    """Arg:
          input_: input tensor of shape [batch_size, input_size]
          output_size: output tensor dim
          
        Return:
          output tensor of shape [batch_size, output_size]
          """
    
    shape = input_.get_shape().as_list()
    stddev = 1. / tf.sqrt(shape[1] / 2.0)
    
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable("Matrix", [shape[1], output_size],
                                 initializer = ly.xavier_initializer(seed = seed))
        bias = tf.get_variable("bias", [output_size], initializer = tf.constant_initializer(0.0))
        
    return tf.matmul(input_, matrix) + bias

def lrelu(x, leak = 0.2, name = 'lrelu'):
    return tf.maximum(x, leak * x)

def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

def batch_norm( x, is_train=True, epsilon = 1e-5, momentum = 0.9, name = "batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay = momentum, updates_collections = None, epsilon = epsilon, scale = True,
                                        is_training = is_train, scope = name)



def flatten(x):
    return tf.contrib.layers.flatten(x)

def gaussian_noise_layer(input_layer, std = 0.5, istrain = True):
    if istrain:
        noise = tf.random_normal(shape = tf.shape(input_layer), mean = 0.0, stddev = std, dtype = tf.float32)
        return input_layer + noise
    else:
        return input_layer

def elu(features):
    return tf.nn.elu(features)

def ResidualBlock(input_, dim, k_h=3, k_w=3, stddev=0.001, name='residual'):
    with tf.variable_scope(name):
        x = relu(batch_norm(input_, name = 'bn0'))
        h1 = conv2d(x, dim, k_h=k_h, k_w=k_w, d_h=1, d_w=1, stddev=stddev, name="conv2d1")
        h1 = relu(batch_norm(h1, name="bn1"))
        y = conv2d(h1, dim, k_h=k_h, k_w=k_w, d_h=1, d_w=1, stddev=stddev, name='conv2d2')
        y = input_ + y
        return y

