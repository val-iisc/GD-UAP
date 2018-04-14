import sys

sys.path.insert(0, 'tensorflow-classification')
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
import time
import utils.functions as func


def validate_arguments(args):
    nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet', 'resnet152']

    if not(args.network in nets):
        print ('invalid network')
        exit(-1)
    if args.adv_im is None:
        print ('no path to perturbation')
        exit(-1)
    if args.img_list is None or args.gt_labels is None:
        print ('provide image list and labels')
        exit(-1)


def choose_net(network, adv_image):
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

    # loading the perturbation
    pert_load = np.load(adv_image)
    # preprocessing if necessary
    if (pert_load.shape[1] == 224 and size == 227):
        pert_load = fff_utils.upsample(np.squeeze(pert_load))
    elif (pert_load.shape[1] == 227 and size == 224):
        pert_load = fff_utils.downsample(np.squeeze(pert_load))
    elif (pert_load.shape[1] not in [224, 227]):
        print(pert_load.shape[1])
        raise Exception("Invalid size of input perturbation")

    adv_image = tf.constant(pert_load, dtype='float32')
    # placeholder to pass image
    input_image = tf.placeholder(
        shape=[None, size, size, 3], dtype='float32', name='input_image')
    input_batch = tf.concat([input_image, tf.add(input_image, adv_image)], 0)

    return MAP[network](input_batch), input_image


def classify(net, in_im, net_name, im_list, gt_labels, batch_size):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    imgs = open(im_list).readlines()  # [::10]
    gt_labels = open(gt_labels).readlines()  # [::10]
    fool_rate = 0
    top_1 = 0
    top_1_real = 0
    isotropic, size = get_params(net_name)
    batch_im = np.zeros((batch_size, size, size, 3))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        img_loader = loader_func(net_name, sess, isotropic, size)
        for i in range(len(imgs)/batch_size):
            lim = min(batch_size, len(imgs)-i*batch_size)
            for j in range(lim):
                im = img_loader(imgs[i*batch_size+j].strip())
                batch_im[j] = np.copy(im)
            gt = np.array([int(gt_labels[i*batch_size+j].strip())
                           for j in range(lim)])
            softmax_scores = sess.run(net['prob'], feed_dict={in_im: batch_im})
            true_predictions = np.argmax(softmax_scores[:batch_size], axis=1)
            ad_predictions = np.argmax(softmax_scores[batch_size:], axis=1)
            if i != 0 and i % 100 == 0:
                print 'iter: {:5d}\ttop-1_real: {:04.2f}\ttop-1: {:04.2f}\tfooling-rate: {:04.2f}'.format(i,
                                                                                                          (top_1_real/float(i*batch_size))*100, (top_1/float(i*batch_size))*100, (fool_rate)/float(i*batch_size)*100)
            top_1 += np.sum(ad_predictions == gt)
            top_1_real += np.sum(true_predictions == gt)
            fool_rate += np.sum(true_predictions != ad_predictions)
    print 'Real Top-1 Accuracy = {:.2f}'.format(
        top_1_real/float(len(imgs))*100)
    print 'Top-1 Accuracy = {:.2f}'.format(top_1/float(len(imgs))*100)
    print 'Fooling Rate = {:.2f}'.format(fool_rate/float(len(imgs))*100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='googlenet',
                        help='The network eg. googlenet')
    parser.add_argument('--adv_im', help='Path to the perturbation image')
    parser.add_argument(
        '--img_list',  help='Path to the validation image list')
    parser.add_argument(
        '--gt_labels', help='Path to the ground truth validation labels')
    parser.add_argument('--batch_size', default=25,
                        help='Batch Size while evaluation.')
    args = parser.parse_args()

    validate_arguments(args)
    net, inp_im = choose_net(args.network, args.adv_im)
    classify(net, inp_im, args.network, args.img_list,
             args.gt_labels, int(args.batch_size))


if __name__ == '__main__':
    main()
