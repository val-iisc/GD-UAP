import numpy as np
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
# util layers

# Obselete
def old_img_preprocess(img_path, size=224):
    mean = [103.939, 116.779, 123.68]
    img = imread(img_path)
    img = resize(img, (size, size))*255.0
    if len(img.shape) == 2:
        img = np.dstack([img,img,img])
    img[:,:,0] -= mean[2]
    img[:,:,1] -= mean[1]
    img[:,:,2] -= mean[0]
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = np.reshape(img,[1,size,size,3])
    return img
# Preprocessing for Inception V3
def v3_preprocess(img_path):
    img = imread(img_path)
    img = resize(img, (299, 299), preserve_range=True)
    img = (img - 128) / 128
    if len(img.shape) == 2:
        img = np.dstack([img,img,img])
    img = np.reshape(img,[1,299,299,3])
    return img

# Image preprocessing format
# Fog VGG models.
def vgg_preprocess(img_path, size=224):
    mean = [103.939, 116.779, 123.68]
    img = imread(img_path)
    if len(img.shape) == 2:
        img = np.dstack([img,img,img])
    resFac = 256.0/min(img.shape[:2])
    newSize = list(map(int, (img.shape[0]*resFac, img.shape[1]*resFac)))
    img = resize(img, newSize, mode='constant', preserve_range=True)
    offset = [newSize[0]/2.0-np.floor(size/2.0), newSize[1]/2.0-np.floor(size/2.0)]
    # print(offset,size)
    img = img[int(offset[0]):int(offset[0])+size, int(offset[1]):int(offset[1])+size, :]
    img[:,:,0] -= mean[2]
    img[:,:,1] -= mean[1]
    img[:,:,2] -= mean[0]
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = np.reshape(img,[1,size,size,3])
    return img

# For Resnets,Caffenet and Googlenet
# From Caffe-tensorflow
def img_preprocess(img, scale=256, isotropic=False, crop=227, mean=np.array([103.939, 116.779, 123.68])):
    '''Crops, scales, and normalizes the given image.
    scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
    crop  : After scaling, a central crop of this size is taken.
    mean  : Subtracted from the image
    '''
    # Rescale
    if isotropic:
        img_shape = tf.to_float(tf.shape(img)[:2])
        min_length = tf.minimum(img_shape[0], img_shape[1])
        new_shape = tf.to_int32((scale / min_length) * img_shape)
    else:
        new_shape = tf.stack([scale, scale])
    img = tf.image.resize_images(img, new_shape)
    # Center crop
    # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
    # See: https://github.com/tensorflow/tensorflow/issues/521
    offset = (new_shape - crop) / 2
    img = tf.slice(img, begin=tf.stack([offset[0], offset[1], 0]), size=tf.stack([crop, crop, -1]))
    # Mean subtraction
    return tf.to_float(img) - mean

def load_image():
    # Read the file
    image_path = tf.placeholder(tf.string,None)
    file_data = tf.read_file(image_path)
    # Decode the image data
    img = tf.image.decode_jpeg(file_data, channels=3)
    img = tf.reverse(img, [-1])
    return img,image_path

def loader_func(network_name,sess,isotropic,size):
    if network_name == 'inceptionv3':
        def loader(image_name):
            im = v3_preprocess(image_name)
            return im
    elif 'vgg' in network_name:
        def loader(image_name):
            im = vgg_preprocess(image_name)
            return im
    else:
        img_tensor,image_path_tensor = load_image()
        processed_img = img_preprocess(img=img_tensor,isotropic=isotropic,crop=size)
        def loader(image_name,processed_img=processed_img,image_path_tensor=image_path_tensor,sess=sess):
            im = sess.run([processed_img],feed_dict={image_path_tensor:image_name})
            return im
    return loader

def get_params(net_name):
    isotropic = False
    if net_name =='caffenet':
        size = 227
    elif net_name=='inceptionv3':
        size = 299
    else:
        size = 224
        if not net_name == 'googlenet':
            isotropic = True
    return isotropic,size
