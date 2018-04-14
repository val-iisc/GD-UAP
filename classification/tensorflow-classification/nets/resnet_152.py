import tensorflow as tf
from misc.layers import *
import numpy as np


def model(image, weights):
    # check image dimensions
    assert image.get_shape().as_list()[1:] == [224, 224, 3]
    layers = {}
    with tf.name_scope("conv1"):
        layers['conv1'] = conv_layer(
            image, weights['conv1']['weights'], s=2, relu=False)
        layers['bn_conv1'] = batch_norm(layers['conv1'], weights['bn_conv1'])
        layers['pool1'] = max_pool(layers['bn_conv1'], k=3, s=2)

    with tf.name_scope("res2"):
        layers['res2a'] = res_block(layers['pool1'], '2a', weights, first=True)
        layers['res2b'] = res_block(layers['res2a'], '2b', weights)
        layers['res2c'] = res_block(layers['res2b'], '2c', weights)

    with tf.name_scope("res3"):
        layers['res3a'] = res_block(
            layers['res2c'], '3a', weights, stride=2, first=True)
        layers['res3b1'] = res_block(layers['res3a'], '3b1', weights)
        layers['res3b2'] = res_block(layers['res3b1'], '3b2', weights)
        layers['res3b3'] = res_block(layers['res3b2'], '3b3', weights)
        layers['res3b4'] = res_block(layers['res3b3'], '3b4', weights)
        layers['res3b5'] = res_block(layers['res3b4'], '3b5', weights)
        layers['res3b6'] = res_block(layers['res3b5'], '3b6', weights)
        layers['res3b7'] = res_block(layers['res3b6'], '3b7', weights)

    with tf.name_scope("res4"):
        layers['res4a'] = res_block(
            layers['res3b7'], '4a', weights, stride=2, first=True)
        layers['res4b1'] = res_block(layers['res4a'], '4b1', weights)
        layers['res4b2'] = res_block(layers['res4b1'], '4b2', weights)
        layers['res4b3'] = res_block(layers['res4b2'], '4b3', weights)
        layers['res4b4'] = res_block(layers['res4b3'], '4b4', weights)
        layers['res4b5'] = res_block(layers['res4b4'], '4b5', weights)
        layers['res4b6'] = res_block(layers['res4b5'], '4b6', weights)
        layers['res4b7'] = res_block(layers['res4b6'], '4b7', weights)
        layers['res4b8'] = res_block(layers['res4b7'], '4b8', weights)
        layers['res4b9'] = res_block(layers['res4b8'], '4b9', weights)
        layers['res4b10'] = res_block(layers['res4b9'], '4b10', weights)
        layers['res4b11'] = res_block(layers['res4b10'], '4b11', weights)
        layers['res4b12'] = res_block(layers['res4b11'], '4b12', weights)
        layers['res4b13'] = res_block(layers['res4b12'], '4b13', weights)
        layers['res4b14'] = res_block(layers['res4b13'], '4b14', weights)
        layers['res4b15'] = res_block(layers['res4b14'], '4b15', weights)
        layers['res4b16'] = res_block(layers['res4b15'], '4b16', weights)
        layers['res4b17'] = res_block(layers['res4b16'], '4b17', weights)
        layers['res4b18'] = res_block(layers['res4b17'], '4b18', weights)
        layers['res4b19'] = res_block(layers['res4b18'], '4b19', weights)
        layers['res4b20'] = res_block(layers['res4b19'], '4b20', weights)
        layers['res4b21'] = res_block(layers['res4b20'], '4b21', weights)
        layers['res4b22'] = res_block(layers['res4b21'], '4b22', weights)
        layers['res4b23'] = res_block(layers['res4b22'], '4b23', weights)
        layers['res4b24'] = res_block(layers['res4b23'], '4b24', weights)
        layers['res4b25'] = res_block(layers['res4b24'], '4b25', weights)
        layers['res4b26'] = res_block(layers['res4b25'], '4b26', weights)
        layers['res4b27'] = res_block(layers['res4b26'], '4b27', weights)
        layers['res4b28'] = res_block(layers['res4b27'], '4b28', weights)
        layers['res4b29'] = res_block(layers['res4b28'], '4b29', weights)
        layers['res4b30'] = res_block(layers['res4b29'], '4b30', weights)
        layers['res4b31'] = res_block(layers['res4b30'], '4b31', weights)
        layers['res4b32'] = res_block(layers['res4b31'], '4b32', weights)
        layers['res4b33'] = res_block(layers['res4b32'], '4b33', weights)
        layers['res4b34'] = res_block(layers['res4b33'], '4b34', weights)
        layers['res4b35'] = res_block(layers['res4b34'], '4b35', weights)

    with tf.name_scope("res5"):
        layers['res5a'] = res_block(
            layers['res4b35'], '5a', weights, stride=2, first=True)
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


def resnet152(input):

    # weigths and biases for tensorflow
    net = np.load('weights/resnet152.npy').item()
    weights = {}
    for name in net.keys():
        weights[name] = {}
        for i in net[name].keys():
            weights[name][i] = tf.Variable(tf.constant(
                net[name][i]), dtype='float32', name=name+'_'+i, trainable=False)

    return model(input, weights)
