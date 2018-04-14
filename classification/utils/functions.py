import numpy as np
import skimage.io as sio
import cv2
import random
import math
import tensorflow as tf
from skimage.transform import resize
from scipy.misc import imread, imresize

# Randomizers


def flip(I, flip_p):
    if flip_p > 0.5:
        return I[:, ::-1, :]
    else:
        return I


def blur(img_temp, blur_p, blur_val):
    if blur_p > 0.5:
        return cv2.GaussianBlur(img_temp, (blur_val, blur_val), 1)
    else:
        return img_temp


def rotate(img_temp, rot, rot_p):
    if(rot_p > 0.5):
        rows, cols, ind = img_temp.shape
        h_pad = int(rows*abs(math.cos(rot/180.0*math.pi)) +
                    cols*abs(math.sin(rot/180.0*math.pi)))
        w_pad = int(cols*abs(math.cos(rot/180.0*math.pi)) +
                    rows*abs(math.sin(rot/180.0*math.pi)))
        final_img = np.zeros((h_pad, w_pad, 3))
        final_img[(h_pad-rows)/2:(h_pad+rows)/2, (w_pad-cols) /
                  2:(w_pad+cols)/2, :] = np.copy(img_temp)
        M = cv2.getRotationMatrix2D((w_pad/2, h_pad/2), rot, 1)
        final_img = cv2.warpAffine(
            final_img, M, (w_pad, h_pad), flags=cv2.INTER_NEAREST)
        part_denom = (math.cos(2*rot/180.0*math.pi))
        w_inside = int((cols*abs(math.cos(rot/180.0*math.pi)) -
                        rows*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        h_inside = int((rows*abs(math.cos(rot/180.0*math.pi)) -
                        cols*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        final_img = final_img[(h_pad-h_inside)/2:(h_pad+h_inside)/2,
                              (w_pad - w_inside)/2:(w_pad + w_inside)/2, :].astype('uint8')
        return final_img
    else:
        return img_temp


def rand_crop(img_temp, dim=224):
    h = img_temp.shape[0]
    w = img_temp.shape[1]
    trig_h = trig_w = False
    if(h > dim):
        h_p = int(random.uniform(0, 1)*(h-dim))
        img_temp = img_temp[h_p:h_p+dim, :, :]
    elif(h < dim):
        trig_h = True
    if(w > dim):
        w_p = int(random.uniform(0, 1)*(w-dim))
        img_temp = img_temp[:, w_p:w_p+dim, :]
    elif(w < dim):
        trig_w = True
    if(trig_h or trig_w):
        pad = np.zeros((dim, dim, 3), dtype='uint8')
        pad[:, :, 0] += 104
        pad[:, :, 1] += 117
        pad[:, :, 2] += 123
        pad[:img_temp.shape[0], :img_temp.shape[1], :] = img_temp
        return pad
    else:
        return img_temp


def randomizer(img_temp):
    dim = 224
    flip_p = random.uniform(0, 1)
    scale_p = random.uniform(0, 1)
    blur_p = random.uniform(0, 1)
    blur_val = random.choice([3, 5, 7, 9])
    rot_p = np.random.uniform(0, 1)
    rot = random.choice([-10, -7, -5, -3, 3, 5, 7, 10])
    if(scale_p > .5):
        scale = random.uniform(0.75, 1.5)
    else:
        scale = 1
    if(img_temp.shape[0] < img_temp.shape[1]):
        ratio = dim*scale/float(img_temp.shape[0])
    else:
        ratio = dim*scale/float(img_temp.shape[1])
    img_temp = cv2.resize(
        img_temp, (int(img_temp.shape[1]*ratio), int(img_temp.shape[0]*ratio)))
    img_temp = flip(img_temp, flip_p)
    img_temp = rotate(img_temp, rot, rot_p)
    img_temp = blur(img_temp, blur_p, blur_val)
    img_temp = rand_crop(img_temp)
    return img_temp


# A generic preprocessor for all the kinds of networks.

def img_preprocess(img_path, size=224, augment=False):
    mean = [103.939, 116.779, 123.68]
    img = imread(img_path)
    if augment:
        img = randomizer(img)
    if len(img.shape) == 2:
        img = np.dstack([img, img, img])
    resFac = 256.0/min(img.shape[:2])
    newSize = list(map(int, (img.shape[0]*resFac, img.shape[1]*resFac)))
    img = resize(img, newSize, mode='constant', preserve_range=True)
    offset = [newSize[0]/2.0 -
              np.floor(size/2.0), newSize[1]/2.0-np.floor(size/2.0)]
    img = img[int(offset[0]):int(offset[0])+size,
              int(offset[1]):int(offset[1])+size, :]
    img[:, :, 0] -= mean[2]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[0]
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = np.reshape(img, [1, size, size, 3])
    return img


def downsample(inp):
    return np.reshape(inp[1:-2, 1:-2, :], [1, 224, 224, 3])


def upsample(inp):
    out = np.zeros([227, 227, 3])
    out[1:-2, 1:-2, :] = inp
    out[0, 1:-2, :] = inp[0, :, :]
    out[-2, 1:-2, :] = inp[-1, :, :]
    out[-1, 1:-2, :] = inp[-1, :, :]
    out[:, 0, :] = out[:, 1, :]
    out[:, -2, :] = out[:, -3, :]
    out[:, -1, :] = out[:, -3, :]
    return np.reshape(out, [1, 227, 227, 3])
