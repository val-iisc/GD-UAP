import tensorflow as tf
from misc.layers import *
import numpy as np


def model(image, weights):
    # check image dimensions
    assert image.get_shape().as_list()[1:] == [224, 224, 3]
    layers = {}
    with tf.name_scope("conv1"):
        layers['conv1'] = conv_layer(
            image, weights['conv1']['weights'], weights['conv1']['biases'], s=2, relu=False)
        layers['bn_conv1'] = batch_norm(layers['conv1'], weights['bn_conv1'])
        layers['pool1'] = max_pool(layers['bn_conv1'], k=3, s=2)

    with tf.name_scope("res2"):
        layers['res2a'] = res_block(layers['pool1'], '2a', weights, first=True)
        layers['res2b'] = res_block(layers['res2a'], '2b', weights)
        layers['res2c'] = res_block(layers['res2b'], '2c', weights)

    with tf.name_scope("res3"):
        layers['res3a'] = res_block(
            layers['res2c'], '3a', weights, stride=2, first=True)
        layers['res3b'] = res_block(layers['res3a'], '3b', weights)
        layers['res3c'] = res_block(layers['res3b'], '3c', weights)
        layers['res3d'] = res_block(layers['res3c'], '3d', weights)

    with tf.name_scope("res4"):
        layers['res4a'] = res_block(
            layers['res3d'], '4a', weights, stride=2, first=True)
        layers['res4b'] = res_block(layers['res4a'], '4b', weights)
        layers['res4c'] = res_block(layers['res4b'], '4c', weights)
        layers['res4d'] = res_block(layers['res4c'], '4d', weights)
        layers['res4e'] = res_block(layers['res4d'], '4e', weights)
        layers['res4f'] = res_block(layers['res4e'], '4f', weights)

    with tf.name_scope("res5"):
        layers['res5a'] = res_block(
            layers['res4f'], '5a', weights, stride=2, first=True)
        layers['res5b'] = res_block(layers['res5a'], '5b', weights)
        layers['res5c'] = res_block(layers['res5b'], '5c', weights)

    with tf.name_scope('pool5'):
        layers['pool5'] = tf.nn.avg_pool(layers['res5c'], [1, 7, 7, 1], [
                                         1, 1, 1, 1], padding='VALID')
        layers['pool5_r'] = tf.reshape(layers['pool5'], [-1, 2048])

    with tf.name_scope('fc1000'):
        layers['fc1000'] = fully_connected(
            layers['pool5_r'], weights['fc1000']['weights'], weights['fc1000']['biases'])
        layers['prob'] = tf.nn.softmax(layers['fc1000'])

    return layers


def resnet50(input):

    # weigths and biases for tensorflow
    net = np.load('weights/resnet50.npy').item()
    weights = {}
    for name in net.keys():
        weights[name] = {}
        for i in net[name].keys():
            weights[name][i] = tf.Variable(tf.constant(
                net[name][i]), dtype='float32', name=name+'_'+i, trainable=False)

    return model(input, weights)
