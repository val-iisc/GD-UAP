import tensorflow as tf
from misc.layers import *
import numpy as np

def model(image, weights, biases, keep_prob=1.0):
    #check image dimensions
    assert image.get_shape().as_list()[1:] == [227, 227, 3]
    layers = {}
    with tf.name_scope("conv1"):
        layers['conv1'] = conv_layer(image, weights['conv1'], biases['conv1'], s=4, padding='VALID')
        layers['pool1'] = max_pool(layers['conv1'], k=3, s=2, padding='VALID')
        layers['norm1'] = tf.nn.lrn(layers['pool1'],2,1.0,2e-05,0.75) 

    with tf.name_scope("conv2"):
        layers['conv2'] = conv_layer(layers['norm1'], weights['conv2'], biases['conv2'], group=2)
        layers['pool2'] = max_pool(layers['conv2'], k=3, s=2, padding='VALID')
        layers['norm2'] = tf.nn.lrn(layers['pool2'],2,1.0,2e-05,0.75) 

    with tf.name_scope("conv3"):
        layers['conv3'] = conv_layer(layers['norm2'], weights['conv3'], biases['conv3'])

    with tf.name_scope("conv4"):
        layers['conv4'] = conv_layer(layers['conv3'], weights['conv4'], biases['conv4'], group=2)

    with tf.name_scope("conv5"):
        layers['conv5'] = conv_layer(layers['conv4'], weights['conv5'], biases['conv5'], group=2)
        layers['pool5'] = max_pool(layers['conv5'], k=3, s=2, padding='VALID')

    flatten = tf.reshape(layers['pool5'], [-1, 9216])

    with tf.name_scope('fc6'):
        layers['fc6'] = tf.nn.relu(fully_connected(flatten, weights['fc6'], biases['fc6']))
        layers['fc6'] = tf.nn.dropout(layers['fc6'], keep_prob=keep_prob)

    with tf.name_scope('fc7'):
        layers['fc7'] = tf.nn.relu(fully_connected(layers['fc6'], weights['fc7'], biases['fc7']))
        layers['fc7'] = tf.nn.dropout(layers['fc7'], keep_prob=keep_prob)

    with tf.name_scope('fc8'):
        layers['fc8'] = fully_connected(layers['fc7'], weights['fc8'], biases['fc8'])
        layers['prob'] = tf.nn.softmax(layers['fc8'])

    return layers

def caffenet(input):

    #weigths and biases for tensorflow
    net = np.load('weights/caffenet.npy').item()
    weights = {}
    biases = {}
    for name in net.keys():
        weights[name] = tf.Variable(tf.constant(net[name]['weights']), dtype='float32' ,name=name+"_weight", trainable=False)
        biases[name] = tf.Variable(tf.constant(net[name]['biases']), dtype='float32' ,name=name+"_bias", trainable=False)

    return model(input, weights, biases)
