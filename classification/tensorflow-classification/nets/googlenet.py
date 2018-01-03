import tensorflow as tf
from misc.layers import *
import numpy as np

def model(image, weights, biases):
    #check image dimensions
    assert image.get_shape().as_list()[1:] == [224, 224, 3]
    layers = {}
    with tf.name_scope("conv1"):
        layers['conv1_7x7_s2'] = conv_layer(image, weights['conv1_7x7_s2'], biases['conv1_7x7_s2'],  s=2)
        layers['pool1_3x3_s2'] = max_pool(layers['conv1_7x7_s2'], k=3, s=2)
        layers['pool1_norm1'] = tf.nn.lrn(layers['pool1_3x3_s2'], 2, 1.0, 2e-05, 0.75)

    with tf.name_scope("conv2"):
        layers['conv2_3x3_reduce'] = conv_layer(layers['pool1_norm1'], weights['conv2_3x3_reduce'], biases['conv2_3x3_reduce'])
        layers['conv2_3x3'] = conv_layer(layers['conv2_3x3_reduce'], weights['conv2_3x3'], biases['conv2_3x3'])
        layers['conv2_norm2'] = tf.nn.lrn(layers['conv2_3x3'], 2, 1.0, 2e-05, 0.75)
        layers['pool2_3x3_s2'] = max_pool(layers['conv2_norm2'], k=3, s=2)

    with tf.name_scope('inception_3'):
        layers['inception_3a_output'] = inception_block(layers['pool2_3x3_s2'], '3a', weights, biases)
        layers['inception_3b_output'] = inception_block(layers['inception_3a_output'], '3b', weights, biases)
        layers['pool3_3x3_s2'] = max_pool(layers['inception_3b_output'], k=3, s=2)

    with tf.name_scope('inception_4'):
        layers['inception_4a_output'] = inception_block(layers['pool3_3x3_s2'], '4a', weights, biases)
        layers['inception_4b_output'] = inception_block(layers['inception_4a_output'], '4b', weights, biases)
        layers['inception_4c_output'] = inception_block(layers['inception_4b_output'], '4c', weights, biases)
        layers['inception_4d_output'] = inception_block(layers['inception_4c_output'], '4d', weights, biases)
        layers['inception_4e_output'] = inception_block(layers['inception_4d_output'], '4e', weights, biases)
        layers['pool4_3x3_s2'] = max_pool(layers['inception_4e_output'], k=3, s=2)

    with tf.name_scope('inception_5'):
        layers['inception_5a_output'] = inception_block(layers['pool4_3x3_s2'], '5a', weights, biases)
        layers['inception_5b_output'] = inception_block(layers['inception_5a_output'], '5b', weights, biases)
        layers['pool5_7x7_s1'] = tf.nn.avg_pool(layers['inception_5b_output'], [1,7,7,1], [1,1,1,1], padding='VALID')
        layers['pool5_7x7_s1'] = tf.reshape(layers['pool5_7x7_s1'], [-1,1024])

    with tf.name_scope('fc'):
        layers['loss3_classifier'] = fully_connected(layers['pool5_7x7_s1'], weights['loss3_classifier'], biases['loss3_classifier'])
        layers['prob'] = tf.nn.softmax(layers['loss3_classifier'])

        return layers

def googlenet(input):

    #weigths and biases for tensorflow
    net = np.load('weights/googlenet.npy').item()
    weights = {}
    biases = {}
    for name in net.keys():
        weights[name] = tf.Variable(tf.constant(net[name]['weights']), dtype='float32' ,name=name+'_weights', trainable=False)
        biases[name] = tf.Variable(tf.constant(net[name]['biases']), dtype='float32' ,name=name+'_biases', trainable=False)

    return model(input, weights, biases)
