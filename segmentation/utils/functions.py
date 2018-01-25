import torch
from torch.autograd import Variable
import random
import time
import torch
import numpy as np
import cv2
import math
from PIL import Image

def proj_lp(v, xi=10.0, p=np.inf):

    # Project on the lp ball centered at 0 and of radius xi
    if p ==np.inf:
            v = torch.clamp(v,-xi,xi)
    else:
        v = v * min(1, xi/(torch.norm(v,p)+0.00001))
    return v


def input_init(xi=10,size=513):
    v = (torch.rand(1,3,size,size)-0.5)*2*xi
    return v

def chunker(seq, size):
    return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list

# Augmentation functions

def flip(I,flip_p):
    if flip_p>0.5:
        return I[:,::-1,:]
    else:
        return I

def blur(img_temp,blur_p):
    if blur_p>0.5:
        return cv2.GaussianBlur(img_temp,(3,3),1)
    else:
        return img_temp

def crop_preprocess(img_temp,dim,net_name):
    h =img_temp.shape[0]
    w = img_temp.shape[1]
    trig_h=trig_w=False
    if(h>dim):
        h_p = int(random.uniform(0,1)*(h-dim))
        img_temp = img_temp[h_p:h_p+dim,:,:]
    elif(h<dim):
        trig_h = True
    if(w>dim):
        w_p = int(random.uniform(0,1)*(w-dim))
        img_temp = img_temp[:,w_p:w_p+dim,:]
    elif(w<dim):
        trig_w = True
    if(trig_h or trig_w):
        pad = np.zeros((dim,dim,3))
        pad[:img_temp.shape[0],:img_temp.shape[1],:] = img_temp
        return pad
    else:
        return img_temp

def crop(img_temp,dim,net_name):
    if net_name in ['dl_resnet_msc','dl_vgg16']:
        h =img_temp.shape[0]
        w = img_temp.shape[1]
        trig_h=trig_w=False
        if(h>dim):
            h_p = int(random.uniform(0,1)*(h-dim))
            img_temp = img_temp[h_p:h_p+dim,:,:]
        elif(h<dim):
            trig_h = True
        if(w>dim):
            w_p = int(random.uniform(0,1)*(w-dim))
            img_temp = img_temp[:,w_p:w_p+dim,:]
        elif(w<dim):
            trig_w = True
        if(trig_h or trig_w):
            pad = np.zeros((dim,dim,3))
            pad[:img_temp.shape[0],:img_temp.shape[1],:] = img_temp
            return pad
        else:
            return img_temp
    else:
        return img_temp

def rotate(img_temp,rot,rot_p):
    if(rot_p>0.5):
        rows,cols,ind = img_temp.shape
        h_pad = int(rows*abs(math.cos(rot/180.0*math.pi)) + cols*abs(math.sin(rot/180.0*math.pi)))
        w_pad = int(cols*abs(math.cos(rot/180.0*math.pi)) + rows*abs(math.sin(rot/180.0*math.pi)))
        final_img = np.zeros((h_pad,w_pad,3))
        final_img[(h_pad-rows)/2:(h_pad+rows)/2,(w_pad-cols)/2:(w_pad+cols)/2,:] = np.copy(img_temp)
        M = cv2.getRotationMatrix2D((w_pad/2,h_pad/2),rot,1)
        final_img = cv2.warpAffine(final_img,M,(w_pad,h_pad),flags = cv2.INTER_NEAREST)
        part_denom = (math.cos(2*rot/180.0*math.pi))
        w_inside = int((cols*abs(math.cos(rot/180.0*math.pi)) - rows*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        h_inside = int((rows*abs(math.cos(rot/180.0*math.pi)) - cols*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        final_img = final_img[(h_pad-h_inside)/2:(h_pad+h_inside)/2,(w_pad- w_inside)/2:(w_pad+ w_inside)/2,:]
        return final_img
    else:
        return img_temp

def randomize(img_temp,dim=513):
    flip_p = random.uniform(0, 1)
    rot_p = random.choice([-10,-7,-5,3,0,3,5,7,10])
    scale_p = random.uniform(0, 1)
    blur_p = random.uniform(0, 1)
    rot = np.random.uniform(0, 1)
    if(scale_p>1):
        scale = random.uniform(1, 1.1)
    else:
        scale = 1
    if(img_temp.shape[0]<img_temp.shape[1]):
        ratio = max(img_temp.shape[0],dim)*scale/float(img_temp.shape[0])
    else:
        ratio = max(img_temp.shape[0],dim)*scale/float(img_temp.shape[1])
    img_temp = cv2.resize(img_temp,(int(img_temp.shape[1]*ratio),int(img_temp.shape[0]*ratio))).astype(float)
    img_temp = flip(img_temp,flip_p)
    img_temp = rotate(img_temp,rot,rot_p)
    img_temp = blur(img_temp,blur_p)
    return img_temp

def img_loader(img_loc,net_name):
    if net_name in ['fcn_alexnet','fcn8s_vgg16']:
        img_temp = Image.open(img_loc)
        img_temp = np.array(img_temp, dtype=np.float32)
        if len(img_temp.shape) == 2:
            img_temp = np.dstack([img_temp,img_temp,img_temp])
        img_temp = np.copy(img_temp[:,:,::-1])
    elif net_name in ['dl_vgg16','dl_resnet_msc']:
        img_temp = cv2.imread(img_loc)
        if len(img_temp.shape) == 2:
            img_temp = np.dstack([img_temp,img_temp,img_temp])
    return img_temp

def get_training_data(chunk,img_path,dim,net_name,randomize=True):
    images = np.zeros((dim,dim,3,len(chunk)))
    for i,piece in enumerate(chunk):
        img_name = piece.split(' ')[0].strip()
        img_temp = img_loader(img_path+img_name,net_name)
        if randomize:
            img_temp = randomize(img_temp)
            
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img_temp = crop(img_temp,dim,net_name)
        images[:,:,:,i] = img_temp
        
    images = images.transpose((3,2,0,1))
    images = Variable(torch.from_numpy(images).float()).cuda()
    return images

def get_testing_data(chunk,im_path,net_name,dim=513):
    images = np.zeros((dim,dim,3,len(chunk)))
    for i,piece in enumerate(chunk):
        img_temp = img_loader(im_path+piece+'.jpg',net_name)
        img_temp = img_temp.astype('float')
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img_temp = crop(img_temp,dim,net_name)
        images[:,:,:,i] = img_temp
        
    images = images.transpose((3,2,0,1))
    images = Variable(torch.from_numpy(images).float(),volatile= True).cuda()
    return images
