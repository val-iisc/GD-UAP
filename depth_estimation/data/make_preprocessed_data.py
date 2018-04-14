'''
This script is for making the substitute dataset from 1000 Places-205 Images.
Preprocessed form of these images are saved for faster evaluation.
'''
import sys
sys.path.insert(0, '../utils')
from functions import *
import numpy as np
import argparse
import os
from skimage.transform import resize
from scipy.misc import imread


def save_preprocessed(save_loc, im_loc, im_list):

    save_name = os.path.join(save_loc, 'preprocessed.npy')

    # check if file already present
    if os.path.isfile(save_name):
        print('Proprocessed Files already exist for this network')
        exit(-1)
    else:
        img_list = open(im_list).readlines()
        size = (256, 512)

        preprocessed_im = np.zeros((1000, size[0], size[1], 3))

        for i in range(1000):
            im_path = os.path.join(im_loc, img_list[i].strip())
            img_temp = imread(im_path)
            if len(img_temp.shape) == 2:
                img_temp = np.stack([img_temp, ]*3, 2)
            img_temp = randomizer(img_temp)
            img_temp = resize(
                img_temp, [256, 512], mode='constant', preserve_range=True)/256.0
            preprocessed_im[i] = np.copy(img_temp)
        np.save(save_name, preprocessed_im)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_loc', default='./',
                        help='location for saving the preprocessed npy.')
    parser.add_argument('--places_im_list', default='../utils/places205_val.txt',
                        help='file containing names of places205 image files')
    parser.add_argument('--places_im_loc', help='location of the image files.')
    args = parser.parse_args()

    save_preprocessed(args.save_loc, args.places_im_loc, args.places_im_list)


if __name__ == '__main__':
    main()
