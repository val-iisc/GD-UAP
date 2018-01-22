import skimage.io as sio
import os
import numpy as np
import cPickle as pickle
import torch
import numpy as np
import time
import os
import math
import scipy.io

def get_file_ids(id_path):
    file_ids = open(id_path)
    ids = []
    for line in file_ids.readlines():
        ids.append(line[:-1])
    return ids

def mat_to_png(location,file_ids,conv_dict):
    ###
    post_fix = '.png'
    ###
    if not os.path.isdir(location+'_converted/'):
        os.makedirs(location+'_converted/')
    for id in file_ids:
        print(os.path.join(location,id+'.mat'))
        img = scipy.io.loadmat(os.path.join(location,id+'_blob_0.mat'))['data']
        img = img.swapaxes(3,0).swapaxes(1,2)[0]
        img = np.argmax(img,0)
        sio.imsave(location+'_converted/'+id+post_fix,img)
        print('Saved:',id+post_fix)
        
def img_dim_reductor(location,file_ids,conv_dict):
    ###
    post_fix = '.png'
    ###
    if not os.path.isdir(location+'_converted/'):
        os.makedirs(location+'_converted/')
    for id in file_ids:
        img = sio.imread(os.path.join(location,id+post_fix))
        img_2d = img_to_2d(img,conv_dict)
        sio.imsave(os.path.join(location+'_converted',id+post_fix),img_2d)
        print('Saved:',id+post_fix)
        
def img_to_2d(img,conv_dict):
    img_2d = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    for key,value in conv_dict.iteritems():
        m = np.all(img==np.array(key).reshape(1, 1, 3), axis=2)
        img_2d[m] = value
    return img_2d

def tensor_maker(cur_batch, predict_loc,gt_loc,gpu=True):
    #returns the tensors of GT and IMG(use 513 and crops)
    #########
    post_fix = '.png'
    #########
    batch_size = len(cur_batch)
    height = 513
    width = 513
    prediction_tensor = np.full((batch_size,height,width),255)
    gt_tensor = np.full((batch_size,height,width),255)
    counter = 0
    for id in cur_batch:
        img = sio.imread(os.path.join(predict_loc,id+post_fix))
        prediction_tensor[counter,:img.shape[0],:img.shape[1]] = np.copy(img)
        img = sio.imread(os.path.join(gt_loc,id+post_fix))
        gt_tensor[counter,:img.shape[0],:img.shape[1]] = np.copy(img)
        counter +=1
    prediction_tensor = torch.from_numpy(prediction_tensor).long()
    gt_tensor = torch.from_numpy(gt_tensor).long()
    if(gpu):
        prediction_tensor = prediction_tensor.cuda()
        gt_tensor = gt_tensor.cuda()
    return (prediction_tensor,gt_tensor)

def hist_per_batch(tensor_1, tensor_2, ignore_label=255, classes=21):
    hist_tensor = torch.zeros(classes,classes)
    for class_2_int in range(classes):
        tensor_2_class = torch.eq(tensor_2,class_2_int).long()
        for class_1_int in range(classes):
            tensor_1_class = torch.eq(tensor_1,class_1_int).long()
            tensor_1_class = torch.mul(tensor_2_class,tensor_1_class)
            count = torch.sum(tensor_1_class)
            hist_tensor[class_2_int,class_1_int] +=count
    return hist_tensor

def hist_maker(predict_loc,gt_loc,file_id_list,batch_size= 20,ignore_label = 255,classes=21,gpu=True):
    hist_tensor = torch.zeros(classes,classes)
    max_iter = int(math.ceil(len(file_id_list)/batch_size)+1)
    for i in range(max_iter):
        cur_batch = file_id_list[batch_size*i:min(len(file_id_list),batch_size*(i+1))]
        predict_tensor, gt_tensor = tensor_maker(cur_batch, predict_loc,gt_loc,gpu = gpu)
        hist_batch = hist_per_batch(predict_tensor, gt_tensor, ignore_label=255, classes=21)
        hist_tensor = torch.add(hist_tensor,hist_batch)
    return hist_tensor

def mean_iou(hist_matrix, class_names):
    classes = len(class_names)
    class_scores = np.zeros((classes))
    for i in range(classes):
        class_scores[i] = hist_matrix[i,i]/(max(1,np.sum(hist_matrix[i,:])+np.sum(hist_matrix[:,i])-hist_matrix[i,i]))
        print('class',class_names[i],'miou',class_scores[i])
    print('Mean IOU:',np.mean(class_scores))
    return class_scores

def mean_pixel_accuracy(hist_matrix, class_names):
    classes = len(class_names)
    class_scores = np.zeros((classes))
    for i in range(classes):
        class_scores[i] = hist_matrix[i,i]/(max(1,np.sum(hist_matrix[i,:])))
        print('class',class_names[i],'mean_pixel_accuracy',class_scores[i])
    return class_scores

def pixel_accuracy(hist_matrix):
    num = np.trace(hist_matrix)
    p_a =  num/max(1,np.sum(hist_matrix).astype('float'))
    print('Pixel accuracy:',p_a)
    return p_a

def freq_weighted_miou(hist_matrix, class_names):
    classes = len(class_names)
    class_scores = np.zeros((classes))
    for i in range(classes):
        class_scores[i] = np.sum(hist_matrix[i,:])*hist_matrix[i,i]/(max(1,np.sum(hist_matrix[i,:])))
    fmiou = np.sum(class_scores)/np.sum(hist_matrix).astype('float')
    print('Frequency Weighted mean accuracy:',fmiou)
    return fmiou
