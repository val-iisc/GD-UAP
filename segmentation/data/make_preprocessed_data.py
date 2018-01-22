'''
This script is for making the substitute dataset from 1000 PASCAL 2012 JPEGImages.
Preprocessed form of these images are saved for faster evaluation.
'''
import sys
sys.path.insert(0,'../utils')
from functions import *

import numpy as np
import argparse
import os
import random

def validate_arguments(args):
    nets = ['fcn_alexnet', 'fcn8s_vgg16', 'dl_vgg16','dl_resnet_msc']
    
    if not(args.network in nets):
        print('invalid network')
        exit(-1)
    

def save_preprocessed(net, save_loc, im_list, im_loc):
    
    save_name_MAP ={
                'fcn_alexnet' : 'fcn_preprocessed.npy',
                'fcn8s_vgg16' : 'fcn_preprocessed.npy',
                'dl_vgg16' : 'dl_preprocessed.npy',
                'dl_resnet_msc' : 'dl_preprocessed.npy',
                 }
    save_name = os.path.join(save_loc,save_name_MAP[net])

    # check if file already present
    if os.path.isfile(save_name):
        print('Proprocessed Files already exist for this network')
        exit(-1)
    else:
        img_list = open(im_list).readlines()
        size = 224
        if net == 'caffenet': size = 227    
        
        preprocessed_im = np.zeros((1000,3,size,size))
        
        for i in range(1000):
            im_path = os.path.normpath(im_loc+img_list[i].strip())
            im = img_loader(im_path,net)
            im = randomize(img)
            img_temp = img_temp.astype('float')
            img_temp[:,:,0] = img_temp[:,:,0] - 104.008
            img_temp[:,:,1] = img_temp[:,:,1] - 116.669
            img_temp[:,:,2] = img_temp[:,:,2] - 122.675
            img_temp = crop(img_temp,dim)
            img_temp = img_temp.tranpose((2,0,1))
            preprocessed_im[i] = np.copy(im)
        np.save(save_name,preprocessed_im)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network',default='all',help="Mention network for which preprocessed data is requried")
    parser.add_argument('--save_loc',default='./',help='location for saving the preprocessed npy.')
    parser.add_argument('--places_im_list',default='../utils/places205_val.txt',help='file containing names of image-files')
    parser.add_argument('--places_im_loc',help='location for the Places 205 folder')
    args = parser.parse_args()
    
    validate_arguments(args)
    save_preprocessed(args.network, args.save_loc,args.places_im_list,args.places_im_loc)

if __name__ == '__main__':
    main()
