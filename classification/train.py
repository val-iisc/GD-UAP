
import sys
sys.path.insert(0, 'tensorflow-classification')

# imports from tensorflow_classification
from nets.vgg_f import vggf
from nets.caffenet import caffenet
from nets.vgg_16 import vgg16
from nets.vgg_19 import vgg19
from nets.googlenet import googlenet
from nets.resnet_152 import resnet152
from misc.utils import *

import tensorflow as tf
import numpy as np
import argparse
import os
import time
import math
import utils.functions as func
import utils.losses as losses


def validate_arguments(args):
    nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet',  'resnet152']

    if not(args.network in nets):
        print ('invalid network')
        exit(-1)


def choose_net(network, train_type):
    MAP = {
        'vggf': vggf,
        'caffenet': caffenet,
        'vgg16': vgg16,
        'vgg19': vgg19,
        'googlenet': googlenet,
        'resnet152': resnet152
    }
    if network == 'caffenet':
        size = 227
    else:
        size = 224
    # placeholder to pass image
    input_image = tf.placeholder(
        shape=[None, size, size, 3], dtype='float32', name='input_image')
    # initializing adversarial image
    adv_image = tf.Variable(tf.random_uniform(
        [1, size, size, 3], minval=-10, maxval=10), name='noise_image', dtype='float32')
    # clipping for imperceptibility constraint
    adv_image = tf.clip_by_value(adv_image, -10, 10)
    input_batch = tf.concat([input_image, tf.add(input_image, adv_image)], 0)
    test_net = MAP[network](input_batch)
    with tf.name_scope("train_net"):
        train_net = MAP[network](tf.add(input_image, adv_image))
    return train_net, test_net, input_image, adv_image


def not_optim_layers(network):
    if network == 'vggf':
        return ['norm1', 'pool1', 'norm2', 'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
    if network == 'caffenet':
        return ['norm1', 'pool1', 'norm2', 'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
    elif network == 'vgg16':
        return ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
    elif network == 'vgg19':
        return ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'prob']
    elif network == 'googlenet':
        return ['pool1_3x3_s2', 'pool1_norm1', 'conv2_norm2', 'pool2_3x3_s2', 'pool3_3x3_s2', 'pool4_3x3_s2', 'pool5_7x7_s1', 'loss3_classifier', 'prob']
    elif network == 'resnet152':
        return ['bn_conv1', 'pool1', 'pool5', 'pool5_r', 'fc1000', 'prob']


def rescale_checker_function(check, sat, sat_change, sat_min):
    value = (sat_change < check and sat > sat_min)
    return value


def get_update_operation_func(train_type, in_im, sess, update, batch_size, size, img_list):
    if train_type == 'no_data':
        def updater(noiser, sess=sess, update=update):
            sess.run(update, feed_dict={in_im: noiser})
    elif train_type == 'with_range':
        def updater(noiser, sess=sess, update=update, in_im=in_im, batch_size=batch_size, size=size):
            image_i = 'data/gaussian_noise.png'
            for j in range(batch_size):
                noiser[j:j+1] = np.copy(func.img_preprocess(image_i,
                                                            size=size, augment=True))
            sess.run(update, feed_dict={in_im: noiser})
    elif train_type == 'with_data':
        def updater(noiser, sess=sess, update=update, in_im=in_im, batch_size=batch_size, size=size, img_list=img_list):
            rander = np.random.randint(low=0, high=(len(img_list)-batch_size))
            for j in range(batch_size):
                noiser[j:j+1] = np.copy(func.img_preprocess(
                    img_list[rander+j].strip(), size=size, augment=True))
            sess.run(update, feed_dict={in_im: noiser})
    return updater


def train(adv_net, net, in_im, ad_im, opt_layers, net_name, train_type, batch_size, img_list_file=None):

    # Vanilla Version
    cost = -losses.l2_all(adv_net, opt_layers)
    tvars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    grads = optimizer.compute_gradients(cost, tvars)
    update = optimizer.apply_gradients(grads)

    size = 224
    # getting the validation set
    if net_name == 'caffenet':
        data_path = os.path.join('data', 'caffenet_preprocessed.npy')
        size = 227
    elif 'vgg' in net_name:
        data_path = os.path.join('data', 'vgg_preprocessed.npy')
    elif net_name == 'googlenet':
        data_path = os.path.join('data', 'googlenet_preprocessed.npy')
    elif net_name == 'resnet152':
        data_path = os.path.join('data', 'resnet_preprocessed.npy')

    imgs = np.load(data_path)
    print('Loaded mini Validation Set')

    # constants
    fool_rate = 0  # current fooling rate
    max_iter = 40000  # better safe than looped into eternity
    stopping = 0  # early stopping condition
    t_s = time.time()
    # New constants
    check = 0.00001
    prev_check = 0
    rescaled = False
    stop_check = False
    noiser = np.zeros((batch_size, size, size, 3))
    rescaled = False
    if train_type == 'with_data':
        img_list = open(img_list_file).readlines()
    else:
        img_list = None

    print "Starting {:} training...".format(net_name)

    # Saturation Measure
    saturation = tf.div(tf.reduce_sum(tf.to_float(
        tf.equal(tf.abs(ad_im), 10))), tf.to_float(tf.size(ad_im)))
    # rate of change of percentage change
    sat_prev = 0
    sat = 0
    sat_change = 0
    sat_min = 0.5  # For checking sat difference only after its sensible

    # rescaler
    assign_op = tvars[0].assign(tf.divide(tvars[0], 2.0))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        update_op = get_update_operation_func(
            train_type, in_im, sess, update, batch_size, size, img_list)
        sess.run(tf.global_variables_initializer())
        for i in range(max_iter):
            update_op(noiser)
            # calculate variables
            sat_prev = np.copy(sat)
            sat = sess.run(saturation)
            sat_change = abs(sat-sat_prev)
            check_dif = i - prev_check
            if i % 100 == 0:
                print('iter', i, 'current_saturation',
                      sat, 'sat_change', sat_change)

            # check for saturation
            if rescale_checker_function(check, sat, sat_change, sat_min):
                rescaled = True
            # validation time
            if not stop_check and ((check_dif > 200 and rescaled == True) or check_dif == 400):
                iters = int(math.ceil(1000/float(batch_size)))
                temp = 0
                prev_check = i
                for j in range(iters):
                    l = j*batch_size
                    L = min((j+1)*batch_size, 1000)
                    softmax_scores = sess.run(
                        net['prob'], feed_dict={in_im: imgs[l:L]})
                    true_predictions = np.argmax(
                        softmax_scores[:batch_size], axis=1)
                    ad_predictions = np.argmax(
                        softmax_scores[batch_size:], axis=1)
                    not_flip = np.sum(true_predictions == ad_predictions)
                    temp += not_flip
                current_rate = (1000-temp)/1000.0
                print('current_val_fooling_rate',
                      current_rate, 'current_iter', i)

                if current_rate > fool_rate:
                    print('best_performance_till_now')
                    stopping = 0
                    fool_rate = current_rate
                    im = sess.run(ad_im)
                    name = 'perturbations/'+net_name+'_'+train_type+'.npy'
                    np.save(name, im)
                else:
                    stopping += 1
                if stopping == 10:
                    print('Val best out')
                    stop_check = True
                    break

            if rescale_checker_function(check, sat, sat_change, sat_min):
                sess.run(assign_op)
                print('reached_saturation', sat, sat_change,
                      'criteria', check, 'iter', i)
                rescaled = False
                prev_check = i
        print('training_done', time.time()-t_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='googlenet',
                        help='The network eg. googlenet')
    parser.add_argument('--prior_type', default='no_data',
                        help='Which kind of prior to use')
    parser.add_argument('--img_list', default='None',
                        help='In case of providing data priors,list of image-files')
    parser.add_argument('--batch_size', default=25,
                        help='The batch size to use for training and testing')
    args = parser.parse_args()
    if args.img_list == 'None':
        args.img_list = None
    validate_arguments(args)
    adv_net, net, inp_im, ad_im = choose_net(args.network, args.prior_type)
    opt_layers = not_optim_layers(args.network)
    train(adv_net, net, inp_im, ad_im, opt_layers, args.network,
          args.prior_type, int(args.batch_size), args.img_list)


if __name__ == '__main__':
    main()
