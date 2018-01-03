import tensorflow as tf
from misc.layers import *
import numpy as np
import h5py

def model(input, weights):
    layers = {}
    layers['conv1'] = conv_layer(input, weights['conv2d_1'], s=2, padding='VALID', relu=False)
    layers['bn1'] = batch_norm(layers['conv1'], weights['batch_normalization_1'])
    layers['conv2'] = conv_layer(layers['bn1'], weights['conv2d_2'], padding='VALID', relu=False)
    layers['bn2'] = batch_norm(layers['conv2'], weights['batch_normalization_2'])
    layers['conv3'] = conv_layer(layers['bn2'], weights['conv2d_3'], relu=False)
    layers['bn3'] = batch_norm(layers['conv3'], weights['batch_normalization_3'])
    layers['pool1'] = max_pool(layers['bn3'], k=3, s=2)

    layers['conv4'] = conv_layer(layers['pool1'], weights['conv2d_4'], padding='VALID', relu=False)
    layers['bn4'] = batch_norm(layers['conv4'], weights['batch_normalization_4'])
    layers['conv5'] = conv_layer(layers['bn4'], weights['conv2d_5'], padding='VALID', relu=False)
    layers['bn5'] = batch_norm(layers['conv5'], weights['batch_normalization_5'])
    layers['pool2'] = max_pool(layers['bn5'], k=3, s=2)

    layers['mixed5b'] = inception_a(layers['pool2'], 'mixed5b', weights, 6)
    layers['mixed5c'] = inception_a(layers['mixed5b']['concat'], 'mixed5c', weights, 13)
    layers['mixed5d'] = inception_a(layers['mixed5c']['concat'], 'mixed5d', weights, 20)

    layers['mixed6a'] = inception_b(layers['mixed5d']['concat'], 'mixed6a', weights, 27)
    layers['mixed6b'] = inception_c(layers['mixed6a']['concat'], 'mixed6b', weights, 31)
    layers['mixed6c'] = inception_c(layers['mixed6b']['concat'], 'mixed6c', weights, 41)
    layers['mixed6d'] = inception_c(layers['mixed6c']['concat'], 'mixed6d', weights, 51)
    layers['mixed6e'] = inception_c(layers['mixed6d']['concat'], 'mixed6e', weights, 61)

    layers['mixed7a'] = inception_d(layers['mixed6e']['concat'], 'mixed7a', weights, 71)
    layers['mixed7b'] = inception_e(layers['mixed7a']['concat'], 'mixed7b', weights, 77)
    layers['mixed7c'] = inception_e(layers['mixed7b']['concat'], 'mixed7c', weights, 86)

    layers['gap'] = avg_pool(layers['mixed7c']['concat'], k=8, padding='VALID')
    layers['gap_r'] = tf.reshape(layers['gap'], [-1,2048])

    layers['classifier'] = fully_connected(layers['gap_r'], weights['predictions']['weights'], weights['predictions']['biases'])
    layers['prob'] = tf.nn.softmax(layers['classifier'])
    return layers

def inceptionv3(input):
    net = h5py.File('weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5','r') 
    weights = {}
    for name in net.keys():
        if 'conv' in name:
            weights[name] = tf.Variable(tf.constant(net[name][name]['kernel:0'][:]), dtype='float32' ,name=name, trainable=False)
        elif 'batch_normalization' in name:
            beta = tf.Variable(tf.constant(net[name][name]['beta:0'][:]), dtype='float32' ,name=name, trainable=False)
            mean = tf.Variable(tf.constant(net[name][name]['moving_mean:0'][:]), dtype='float32' ,name=name, trainable=False)
            variance = tf.Variable(tf.constant(net[name][name]['moving_variance:0'][:]), dtype='float32' ,name=name, trainable=False)
            weights[name] = {'offset': beta, 'mean': mean, 'variance': variance, 'scale': None}
        elif name == 'predictions':
            param = tf.Variable(tf.constant(net[name][name]['kernel:0'][:]), dtype='float32' ,name=name, trainable=False)
            bias = tf.Variable(tf.constant(net[name][name]['bias:0'][:]), dtype='float32' ,name=name, trainable=False)
            weights[name] = {'biases': bias, 'weights': param}

    return model(input, weights)
