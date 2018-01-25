
import numpy as np
import argparse
import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from nets.deeplab_resnet import Res_Deeplab
from nets.deeplab_large_fov import deeplab_vgg_lfov
from nets.fcn_alexnet import fcn_alexnet
from nets.fcn8s_vgg16 import fcn8s_vgg16
from utils.functions import *

def validate_arguments(args):
    nets = ['dl_vgg16', 'dl_resnet_msc','fcn_alexnet','fcn8s_vgg16']

    if not(args.network in nets):
        print ('invalid network')
        exit (-1)

def choose_net(network):
    MAP = {
            'dl_vgg16'     : [deeplab_vgg_lfov,'weights/VOC12_deeplab_1_6000.pth'],
            'fcn8s_vgg16'     : [fcn8s_vgg16,'weights/fcn8s_vgg16.pth'],
            'fcn_alexnet' : [fcn_alexnet,'weights/fcn_alexnet.pth'],
            'dl_resnet_msc' : [Res_Deeplab,'weights/MS_DeepLab_resnet_trained_VOC.pth']
    }
    size = 513
    net = MAP[network][0]()
    for params in net.parameters():
        requires_grad = False
    net.load_state_dict(torch.load(MAP[network][1]))
    net.eval()
    net.cuda()
    return net,size

def rescale_checker_function(check,sat,sat_change,sat_min):
    value = (sat_change<check and sat>sat_min)
    return value

def set_hooks(model_name,model):
    
    def get_norm(self, forward_input, forward_output):
        global main_value
        main_value[0] = main_value[0] -torch.log((torch.norm(forward_output)))
    
    layers_to_opt = get_layers_to_opt(model_name,model)
    print('Optimizing at the following layers')
    print(layers_to_opt)
    for name,layer in model.named_modules():
        if(name in layers_to_opt):
            layer.register_forward_hook(get_norm)
    return model
    
def get_layers_to_opt(model_name,model):
    layers_to_opt= []
    if model_name in ['dl_vgg16','fcn8s_vgg16','fcn_alexnet']:
        for name,layer in model.named_modules():
            if('relu' in name) or ('interp_test' in name)or ('upscore_final' in name):
                layers_to_opt.append(name)
        
    elif model_name == 'dl_resnet_msc':
        for name,layer in model.named_modules():
            if('relu' in name):
                layers_to_opt.append(name)
                
    return layers_to_opt

def train(model,size,net_name,train_batch_size,val_batch_size,prior_type,img_list,img_path):
    
    init_v = input_init()
    v = torch.autograd.Variable(init_v.cuda(),requires_grad=True)
    
    ## The Loss
    
    global main_value
    main_value = [0]
    main_value[0] =torch.autograd.Variable(torch.zeros(1)).cuda()
    
    model = set_hooks(net_name,model)
    
    optimer = optim.Adam([v], lr = 0.1)
    
    # getting the validation set
    if (net_name in  ['fcn8s_vgg16','fcn_alexnet']):
        data_path =os.path.join('data','fcn_preprocessed.npy')
    elif net_name in ['dl_vgg16','dl_resnet_msc']:
        data_path = os.path.join('data','dl_preprocessed.npy')
    imgs = np.load(data_path)
    imgs = torch.FloatTensor(imgs)
    
    print('Loaded mini Validation Set')
    ## constants
    fool_rate = 0 # current fooling rate
    max_iter = 40000
    stopping = 0 # early stopping condition
    t_s = time.time()
    ### New constants
    check = 0.00001
    prev_check = 0
    rescaled = False
    stop_check= False
    sat_prev = 0
    sat = 0
    sat_change = 0
    sat_min=0.5
    if prior_type == 'with_data':
        img_list = open(img_list_file).readlines()
        num_images = len(img_list)
    else:
        img_list = None

    print "Starting {:} training...".format(net_name)
    for i in range(max_iter):
        
        optimer.zero_grad()
        
        # for big batch case
        if prior_type == 'with_data':
            rander = np.random.randint(low=0,high=(imgs.size()[0]-train_batch_size))
            images = get_training_data(img_list[rander:rander+train_batch_size],img_path,size,net_name)
            inp = images+torch.stack((v[0],)*(train_batch_size),0)
            out = model(inp)
        elif prior_type == 'with_range':
            img = 'data/gaussian_noise.png'
            img_list = [img,]*train_batch_size
            images = get_data_from_chunk_v2_noise(img_list,'',size)
            inp = images+torch.stack((v[0],)*(train_batch_size),0)
            out = model(inp)
        elif prior_type =='no_data':
            out = model(v)

        # Update_operation
        loss = main_value[0] 
        loss.backward()
        optimer.step()
        main_value[0] = torch.autograd.Variable(torch.zeros(1)).cuda()
        v.data = proj_lp(v.data)
        
        # Checker for Rescale
        sat_prev = np.copy(sat)
        siim = torch.eq(torch.abs(v.data),10.0).float()
        sim = torch.sum(siim)
        sat = sim/float(torch.numel(v.data))
        sat_change = abs(sat-sat_prev)
        check_dif = i -prev_check
        if rescale_checker_function(check,sat,sat_change,sat_min):
             rescaled =True

        if i%100==0:
            print('iter',i,'current_saturation',sat,'sat_change',sat_change)

        # validation time
        if not stop_check and ((check_dif>200 and rescaled == True) or check_dif==400):
            iters = int(math.ceil(1000/float(val_batch_size)))
            temp = 0
            prev_check = i
            # Make the forward faster
            for param in model.parameters():
                param.volatile = True
                param.requires_grad = False

            cur_batch = torch.autograd.Variable(torch.zeros(val_batch_size,3,513,513)).cuda()
            for j in range(iters):
                l = j*val_batch_size
                L = min((j+1)*val_batch_size,999)
                cur = imgs[l:L,:,:size,:size]
                cur_batch.data = cur.cuda()
                out_normal = model.forward(cur_batch)#[3]
                out_normal = out_normal#.cpu().data.numpy()
                cur_batch.data = cur.cuda()+torch.stack((v[0].data,)*(L-l),0)
                out_pert = model.forward(cur_batch)#[3]
                out_pert = out_pert#.cpu().data.numpy()
                _,true_predictions = torch.max(out_normal,1)#.cpu().data.numpy()
                _,ad_predictions = torch.max(out_pert,1)#.cpu().data.numpy()
                not_flip = torch.sum(torch.eq(true_predictions,ad_predictions).float()).cpu().data.numpy()
                temp += (val_batch_size*size*size - not_flip)/float(val_batch_size*size*size)
            current_rate = temp/float(iters)
            print('current_per_pixel_flipping_rate', current_rate,'current_iter',i)
            if current_rate>fool_rate:
                print('best_performance_till_now')
                stopping =0
                fool_rate = current_rate
                im = v.cpu().data.numpy()
                name = os.path.join('perturbations','new_'+net_name+'_'+prior_type+'.npy')
                np.save(name,im)
            elif current_rate ==fool_rate:
                im = v.cpu().data.numpy()
                name = os.path.join('perturbations','new_'+net_name+'_'+prior_type+'.npy')
                np.save(name,im)
            else:
                stopping+=1
            if stopping==10:
                print('Val best out')
                stop_check =True
                break
            for param in model.parameters():
                param.volatile = False
                param.requires_grad = False
            v.volatile=False
            v.requires_grad=True
            main_value[0] = torch.autograd.Variable(torch.zeros(1)).cuda()     

        if rescale_checker_function(check,sat,sat_change,sat_min):
            v.data = torch.div(v.data,2.0)
            print('reached_saturation',sat,sat_change,'criteria',check,'iter',i)
            rescaled = False
            prev_check = i
    print('training_done', time.time()-t_s)
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='dl_vgg16', help='The network eg. Deeplab Large FOV VGG16')
    parser.add_argument('--prior_type', default='no_data', help='Which kind of prior to use')
    parser.add_argument('--train_batch_size', default=1, help='The batch size to use for training.')
    parser.add_argument('--val_batch_size', default=10, help='The batch size to use for validation.')
    parser.add_argument('--im_path',help='The location of the training images.')
    parser.add_argument('--im_list', default='None', help='In case of providing data priors,list of image-files')
    parser.add_argument('--gpu', default='0', help='The ID of GPU to use.')
    args = parser.parse_args()
    torch.cuda.set_device(int(args.gpu))
    if args.im_list == 'None':
        args.im_list = None
    #args.network = 'vgg16'
    validate_arguments(args)
    net,size  = choose_net(args.network)
    train(net,size,args.network,args.train_batch_size,args.val_batch_size,
         args.prior_type,args.im_list,args.im_path)
if __name__ == '__main__':
    main()
