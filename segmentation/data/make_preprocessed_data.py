'''
This script is for making the substitute dataset from 1000 Imagenet Images.
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
    

def save_preprocessed(net, save_loc, im_list):
    
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
        size = 513
        
        preprocessed_im = np.zeros((1000,3,size,size))
        print('Saving the preprocessed Blob ...')
        
        for i in range(1000):
            im_path = img_list[i].strip()
            img_temp = img_loader(im_path,net)
            im_temp = randomizer(img_temp)
            # To avoid to much padding
            ratio = 2
            img_temp = cv2.resize(img_temp,(int(img_temp.shape[1]*ratio),int(img_temp.shape[0]*ratio))).astype(float)
            img_temp = img_temp.astype('float')
            img_temp[:,:,0] = img_temp[:,:,0] - 104.008
            img_temp[:,:,1] = img_temp[:,:,1] - 116.669
            img_temp[:,:,2] = img_temp[:,:,2] - 122.675
            img_temp = crop_train(img_temp,size,net)
            img_temp = img_temp.transpose((2,0,1))
            preprocessed_im[i] = np.copy(img_temp)
            if i%100 == 0 : print('Current Iteration: ',i)
        np.save(save_name,preprocessed_im)
        print('Preprocessed Blob Saved.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network',default='fcn_alexnet',help="Mention network for which preprocessed data is requried")
    parser.add_argument('--save_loc',default='./',help='location for saving the preprocessed npy.')
    parser.add_argument('--ilsvrc_im_list',default='../utils/ilsvrc_val.txt',help='file containing names of image-files')
    args = parser.parse_args()
    
    validate_arguments(args)
    save_preprocessed(args.network, args.save_loc,args.ilsvrc_im_list)

if __name__ == '__main__':
    main()
