'''
wrapper functions for tensorflow layers
'''
import tensorflow as tf


def conv_layer(bottom, weight, bias=None, s=1, padding='SAME', relu=True, group=1):
    if group == 1:
        conv = tf.nn.conv2d(bottom, weight, [1, s, s, 1], padding=padding)
    else:
        input_split = tf.split(bottom, group, 3)
        weight_split = tf.split(weight, group, 3)
        conv_1 = tf.nn.conv2d(input_split[0], weight_split[0], [
                              1, s, s, 1], padding=padding)
        conv_2 = tf.nn.conv2d(input_split[1], weight_split[1], [
                              1, s, s, 1], padding=padding)
        conv = tf.concat([conv_1, conv_2], 3)
    if bias is None:
        if relu:
            return tf.nn.relu(conv)
        else:
            return conv
    else:
        bias = tf.nn.bias_add(conv, bias)
        if relu:
            return tf.nn.relu(bias)
        else:
            return bias


def reluer(bottom):
    return tf.nn.relu(bottom)


def max_pool(bottom, k=3, s=1, padding='SAME'):
    return tf.nn.max_pool(bottom, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding)


def avg_pool(bottom, k=3, s=1, padding='SAME'):
    return tf.nn.avg_pool(bottom, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding)


def fully_connected(bottom, weight, bias):
    fc = tf.nn.bias_add(tf.matmul(bottom, weight), bias)
    return fc


def batch_norm(bottom, weight, relu=True):
    bn = tf.nn.batch_normalization(
        bottom, weight['mean'], weight['variance'], weight['offset'], weight['scale'], 1e-5)
    if relu:
        return tf.nn.relu(bn)
    else:
        return bn


def inception_block(bottom, name, weights, biases):
    block = {}
    with tf.name_scope(name+'1x1'):
        block['branch_1'] = conv_layer(
            bottom, weights['inception_'+name+'_1x1'], biases['inception_'+name+'_1x1'])

    with tf.name_scope(name+'3x3'):
        block['branch_2_r'] = conv_layer(
            bottom, weights['inception_'+name+'_3x3_reduce'], biases['inception_'+name+'_3x3_reduce'])
        block['branch_2'] = conv_layer(
            block['branch_2_r'], weights['inception_'+name+'_3x3'], biases['inception_'+name+'_3x3'])

    with tf.name_scope(name+'5x5'):
        block['branch_3_r'] = conv_layer(
            bottom, weights['inception_'+name+'_5x5_reduce'], biases['inception_'+name+'_5x5_reduce'])
        block['branch_3'] = conv_layer(
            block['branch_3_r'], weights['inception_'+name+'_5x5'], biases['inception_'+name+'_5x5'])

    with tf.name_scope(name+'pool'):
        block['branch_4_p'] = max_pool(bottom)
        block['branch_4'] = conv_layer(
            block['branch_4_p'], weights['inception_'+name+'_pool_proj'], biases['inception_'+name+'_pool_proj'])

    block['concat'] = tf.concat(axis=3, values=[
                                block['branch_1'], block['branch_2'], block['branch_3'], block['branch_4']])

    return block['concat']


def inception_a(bottom, name, weights, index):
    block = {}
    with tf.name_scope(name+'1x1'):
        block['branch_1'] = conv_layer(
            bottom, weights['conv2d_'+str(index)], relu=False)
        block['branch_1'] = batch_norm(
            block['branch_1'], weights['batch_normalization_'+str(index)])
    with tf.name_scope(name+'5x5'):
        block['branch_2'] = conv_layer(
            bottom, weights['conv2d_'+str(index+1)], relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+1)])
        block['branch_2'] = conv_layer(
            block['branch_2'], weights['conv2d_'+str(index+2)], relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+2)])
    with tf.name_scope(name+'5x5'):
        block['branch_3'] = conv_layer(
            bottom, weights['conv2d_'+str(index+3)], relu=False)
        block['branch_3'] = batch_norm(
            block['branch_3'], weights['batch_normalization_'+str(index+3)])
        block['branch_3'] = conv_layer(
            block['branch_3'], weights['conv2d_'+str(index+4)], relu=False)
        block['branch_3'] = batch_norm(
            block['branch_3'], weights['batch_normalization_'+str(index+4)])
        block['branch_3'] = conv_layer(
            block['branch_3'], weights['conv2d_'+str(index+5)], relu=False)
        block['branch_3'] = batch_norm(
            block['branch_3'], weights['batch_normalization_'+str(index+5)])
    with tf.name_scope(name+'pool'):
        block['branch_4'] = avg_pool(bottom)
        block['branch_4'] = conv_layer(
            block['branch_4'], weights['conv2d_'+str(index+6)], relu=False)
        block['branch_4'] = batch_norm(
            block['branch_4'], weights['batch_normalization_'+str(index+6)])
    block['concat'] = tf.concat(axis=3, values=[
                                block['branch_1'], block['branch_2'], block['branch_3'], block['branch_4']])
    return block


def inception_b(bottom, name, weights, index):
    block = {}
    with tf.name_scope(name+'1'):
        block['branch_1'] = conv_layer(
            bottom, weights['conv2d_'+str(index)], s=2, padding='VALID', relu=False)
        block['branch_1'] = batch_norm(
            block['branch_1'], weights['batch_normalization_'+str(index)])
    with tf.name_scope(name+'2'):
        block['branch_2'] = conv_layer(
            bottom, weights['conv2d_'+str(index+1)], relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+1)])
        block['branch_2'] = conv_layer(
            block['branch_2'], weights['conv2d_'+str(index+2)], relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+2)])
        block['branch_2'] = conv_layer(
            block['branch_2'], weights['conv2d_'+str(index+3)], s=2, padding='VALID', relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+3)])
    with tf.name_scope(name+'pool'):
        block['branch_3'] = max_pool(bottom, s=2, padding='VALID')
    block['concat'] = tf.concat(
        axis=3, values=[block['branch_1'], block['branch_2'], block['branch_3']])
    return block


def inception_c(bottom, name, weights, index):
    block = {}
    with tf.name_scope(name+'1'):
        block['branch_1'] = conv_layer(
            bottom, weights['conv2d_'+str(index)], relu=False)
        block['branch_1'] = batch_norm(
            block['branch_1'], weights['batch_normalization_'+str(index)])
    with tf.name_scope(name+'2'):
        block['branch_2'] = conv_layer(
            bottom, weights['conv2d_'+str(index+1)], relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+1)])
        block['branch_2'] = conv_layer(
            block['branch_2'], weights['conv2d_'+str(index+2)], relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+2)])
        block['branch_2'] = conv_layer(
            block['branch_2'], weights['conv2d_'+str(index+3)], relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+3)])
    with tf.name_scope(name+'3'):
        block['branch_3'] = conv_layer(
            bottom, weights['conv2d_'+str(index+4)], relu=False)
        block['branch_3'] = batch_norm(
            block['branch_3'], weights['batch_normalization_'+str(index+4)])
        block['branch_3'] = conv_layer(
            block['branch_3'], weights['conv2d_'+str(index+5)], relu=False)
        block['branch_3'] = batch_norm(
            block['branch_3'], weights['batch_normalization_'+str(index+5)])
        block['branch_3'] = conv_layer(
            block['branch_3'], weights['conv2d_'+str(index+6)], relu=False)
        block['branch_3'] = batch_norm(
            block['branch_3'], weights['batch_normalization_'+str(index+6)])
        block['branch_3'] = conv_layer(
            block['branch_3'], weights['conv2d_'+str(index+7)], relu=False)
        block['branch_3'] = batch_norm(
            block['branch_3'], weights['batch_normalization_'+str(index+7)])
        block['branch_3'] = conv_layer(
            block['branch_3'], weights['conv2d_'+str(index+8)], relu=False)
        block['branch_3'] = batch_norm(
            block['branch_3'], weights['batch_normalization_'+str(index+8)])
    with tf.name_scope(name+'pool'):
        block['branch_4'] = avg_pool(bottom)
        block['branch_4'] = conv_layer(
            block['branch_4'], weights['conv2d_'+str(index+9)], relu=False)
        block['branch_4'] = batch_norm(
            block['branch_4'], weights['batch_normalization_'+str(index+9)])
    block['concat'] = tf.concat(axis=3, values=[
                                block['branch_1'], block['branch_2'], block['branch_3'], block['branch_4']])
    return block


def inception_d(bottom, name, weights, index):
    block = {}
    with tf.name_scope(name+'1'):
        block['branch_1'] = conv_layer(
            bottom, weights['conv2d_'+str(index)], relu=False)
        block['branch_1'] = batch_norm(
            block['branch_1'], weights['batch_normalization_'+str(index)])
        block['branch_1'] = conv_layer(
            block['branch_1'], weights['conv2d_'+str(index+1)], s=2, padding='VALID', relu=False)
        block['branch_1'] = batch_norm(
            block['branch_1'], weights['batch_normalization_'+str(index+1)])
    with tf.name_scope(name+'2'):
        block['branch_2'] = conv_layer(
            bottom, weights['conv2d_'+str(index+2)], relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+2)])
        block['branch_2'] = conv_layer(
            block['branch_2'], weights['conv2d_'+str(index+3)], relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+3)])
        block['branch_2'] = conv_layer(
            block['branch_2'], weights['conv2d_'+str(index+4)], relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+4)])
        block['branch_2'] = conv_layer(
            block['branch_2'], weights['conv2d_'+str(index+5)], s=2, padding='VALID', relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+5)])
    with tf.name_scope(name+'pool'):
        block['branch_3'] = max_pool(bottom, s=2, padding='VALID')
    block['concat'] = tf.concat(
        axis=3, values=[block['branch_1'], block['branch_2'], block['branch_3']])
    return block


def inception_e(bottom, name, weights, index):
    block = {}
    with tf.name_scope(name+'1'):
        block['branch_1'] = conv_layer(
            bottom, weights['conv2d_'+str(index)], relu=False)
        block['branch_1'] = batch_norm(
            block['branch_1'], weights['batch_normalization_'+str(index)])
    with tf.name_scope(name+'2'):
        block['branch_2'] = conv_layer(
            bottom, weights['conv2d_'+str(index+1)], relu=False)
        block['branch_2'] = batch_norm(
            block['branch_2'], weights['batch_normalization_'+str(index+1)])
        block['branch_2a'] = conv_layer(
            block['branch_2'], weights['conv2d_'+str(index+2)], relu=False)
        block['branch_2a'] = batch_norm(
            block['branch_2a'], weights['batch_normalization_'+str(index+2)])
        block['branch_2b'] = conv_layer(
            block['branch_2'], weights['conv2d_'+str(index+3)], relu=False)
        block['branch_2b'] = batch_norm(
            block['branch_2b'], weights['batch_normalization_'+str(index+3)])
        block['branch_2'] = tf.concat(
            axis=3, values=[block['branch_2a'], block['branch_2b']])
    with tf.name_scope(name+'3'):
        block['branch_3'] = conv_layer(
            bottom, weights['conv2d_'+str(index+4)], relu=False)
        block['branch_3'] = batch_norm(
            block['branch_3'], weights['batch_normalization_'+str(index+4)])
        block['branch_3'] = conv_layer(
            block['branch_3'], weights['conv2d_'+str(index+5)], relu=False)
        block['branch_3'] = batch_norm(
            block['branch_3'], weights['batch_normalization_'+str(index+5)])
        block['branch_3a'] = conv_layer(
            block['branch_3'], weights['conv2d_'+str(index+6)], relu=False)
        block['branch_3a'] = batch_norm(
            block['branch_3a'], weights['batch_normalization_'+str(index+6)])
        block['branch_3b'] = conv_layer(
            block['branch_3'], weights['conv2d_'+str(index+7)], relu=False)
        block['branch_3b'] = batch_norm(
            block['branch_3b'], weights['batch_normalization_'+str(index+7)])
        block['branch_3'] = tf.concat(
            axis=3, values=[block['branch_3a'], block['branch_3b']])
    with tf.name_scope(name+'pool'):
        block['branch_4'] = avg_pool(bottom)
        block['branch_4'] = conv_layer(
            block['branch_4'], weights['conv2d_'+str(index+8)], relu=False)
        block['branch_4'] = batch_norm(
            block['branch_4'], weights['batch_normalization_'+str(index+8)])
    block['concat'] = tf.concat(axis=3, values=[
                                block['branch_1'], block['branch_2'], block['branch_3'], block['branch_4']])
    return block


def res_block(bottom, name, weights, stride=1, first=False):
    with tf.name_scope(name+'_a'):
        c1 = conv_layer(
            bottom, weights['res'+name+'_branch2a']['weights'], s=stride, relu=False)
        bn1 = batch_norm(c1, weights['bn'+name+'_branch2a'])

    with tf.name_scope(name+'_b'):
        c2 = conv_layer(
            bn1, weights['res'+name+'_branch2b']['weights'], relu=False)
        bn2 = batch_norm(c2, weights['bn'+name+'_branch2b'])

    with tf.name_scope(name+'_c'):
        c3 = conv_layer(
            bn2, weights['res'+name+'_branch2c']['weights'], relu=False)
        bn3 = batch_norm(c3, weights['bn'+name+'_branch2c'], relu=False)

    if first:
        with tf.name_scope(name+'_1'):
            c4 = conv_layer(
                bottom, weights['res'+name+'_branch1']['weights'], s=stride, relu=False)
            bn4 = batch_norm(c4, weights['bn'+name+'_branch1'], relu=False)
        return tf.nn.relu(tf.add(bn4, bn3))
    else:
        return tf.nn.relu(tf.add(bottom, bn3))
