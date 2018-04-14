'''
This script is for making the substitute dataset from 1000 PASCAL 2012 JPEGImages.
Preprocessed form of these images are saved for faster evaluation.
'''
import sys
sys.path.insert(0, '../tensorflow-classification')
from misc.utils import *

import tensorflow as tf
import numpy as np
import argparse
import os
import random


def validate_arguments(args):
    nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet', 'resnet152']

    if not(args.network in nets):
        print('invalid network')
        exit(-1)


def save_preprocessed(net, save_loc, im_list, im_loc):

    save_name_MAP = {
        'vggf': 'vgg_preprocessed.npy',
        'vgg16': 'vgg_preprocessed.npy',
                'vgg19': 'vgg_preprocessed.npy',
                'googlenet': 'googlenet_preprocessed.npy',
                'resnet152': 'resnet_preprocessed.npy',
                'caffenet': 'caffenet_preprocessed.npy'
    }
    save_name = os.path.join(save_loc, save_name_MAP[net])

    # check if file already present
    if os.path.isfile(save_name):
        print('Proprocessed Files already exist for this network')
        exit(-1)
    else:
        img_list = open(im_list).readlines()
        size = 224
        if net == 'caffenet':
            size = 227

        preprocessed_im = np.zeros((1000, size, size, 3))

        isotropic, size = get_params(net)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        img_loader = loader_func(net, sess, isotropic, size)
        random.shuffle(img_list)
        for i in range(1000):
            im_path = os.path.normpath(im_loc+img_list[i].strip())
            im = img_loader(im_path)
            preprocessed_im[i] = np.copy(im)
        np.save(save_name, preprocessed_im)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='all',
                        help="Mention network for which preprocessed data is requried")
    parser.add_argument('--save_loc', default='./',
                        help='location for saving the preprocessed npy.')
    parser.add_argument('--pascal_im_list', default='../utils/pascal_val.txt',
                        help='file containing names of image-files')
    parser.add_argument('--pascal_im_loc',
                        help='location for the pascal VOC 2012 folder')
    args = parser.parse_args()

    validate_arguments(args)
    save_preprocessed(args.network, args.save_loc,
                      args.pascal_im_list, args.pascal_im_loc)


if __name__ == '__main__':
    main()
