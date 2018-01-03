from nets.vgg_f import vggf
from nets.caffenet import caffenet
from nets.vgg_16 import vgg16
from nets.vgg_19 import vgg19
from nets.googlenet import googlenet
from nets.resnet_50 import resnet50
from nets.resnet_152 import resnet152
from nets.inception_v3 import inceptionv3
from misc.utils import *
import tensorflow as tf
import numpy as np
import argparse
import time

def validate_arguments(args):
    nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet', 'resnet50', 'resnet152', 'inceptionv3']
    if not(args.network in nets):
        print ('invalid network')
        exit (-1)
    if args.evaluate:
        if args.img_list is None or args.gt_labels is None:
            print ('provide image list and labels')
            exit (-1)

def choose_net(network):    
    MAP = {
        'vggf'     : vggf,
        'caffenet' : caffenet,
        'vgg16'    : vgg16,
        'vgg19'    : vgg19, 
        'googlenet': googlenet, 
        'resnet50' : resnet50,
        'resnet152': resnet152, 
        'inceptionv3': inceptionv3,
    }
    
    if network == 'caffenet':
        size = 227
    elif network == 'inceptionv3':
        size = 299
    else:
        size = 224
        
    #placeholder to pass image
    input_image = tf.placeholder(shape=[None, size, size, 3],dtype='float32', name='input_image')

    return MAP[network](input_image), input_image

def evaluate(net, im_list, in_im, labels, net_name,batch_size=30):
    top_1 = 0
    top_5 = 0
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    img_list = open(im_list).readlines()
    gt_labels = open(labels).readlines()
    t_s = time.time()
    isotropic,size = get_params(net_name)
    batch_im = np.zeros((batch_size, size,size,3))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        img_loader = loader_func(net_name,isotropic,size,sess)
        for i in range(len(img_list)/batch_size):
            lim = min(batch_size,len(img_list)-i*batch_size)
            for j in range(lim):
                im = img_loader(img_list[i*batch_size+j].strip())
                batch_im[j] = np.copy(im)
            gt = np.array([int(gt_labels[i*batch_size+j].strip()) for j in range(lim)])
            softmax_scores = sess.run(net['prob'], feed_dict={in_im: batch_im})
            inds = np.argsort(softmax_scores, axis=1)[:,::-1][:,:5]
            if i!=0 and (i*batch_size+lim)%1000 == 0:
                print 'iter: {:5d}\ttop-1: {:04.2f}\ttop-5: {:04.2f}'.format(i*batch_size+lim, (top_1/float(i*batch_size+lim))*100,
                                                                             (top_5)/float(i*batch_size+lim)*100)
            top_1+= np.sum(inds[:,0] == gt)
            top_5 += np.sum([gt[i] in inds[i] for i in range(lim)])
    print 'Top-1 Accuracy = {:.2f}'.format(top_1/500.0)
    print 'Top-5 Accuracy = {:.2f}'.format(top_5/500.0)
    print 'Time taken: {:.2f}s'.format(time.time()-t_s)
    
def predict(net, im_path, in_im, net_name):
    synset = open('misc/ilsvrc_synsets.txt').readlines()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    t_s = time.time()
    isotropic,size = get_params(net_name)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        img_loader = loader_func(net_name,isotropic,size,sess)
        im = img_loader(im_path.strip())
        im = np.reshape(im,[1,size,size,3])
        softmax_scores = sess.run(net['prob'], feed_dict={in_im: im})
        inds = np.argsort(softmax_scores[0])[::-1][:5]
        print '{:}\t{:}'.format('Score','Class')
        for i in inds:
            print '{:.4f}\t{:}'.format(softmax_scores[0,i], synset[i].strip().split(',')[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='googlenet', help='The network eg. googlenet')
    parser.add_argument('--img_path', default='misc/sample.jpg',  help='Path to input image')
    parser.add_argument('--evaluate', default=False,  help='Flag to evaluate over full validation set')
    parser.add_argument('--img_list',  help='Path to the validation image list')
    parser.add_argument('--gt_labels', help='Path to the ground truth validation labels')
    parser.add_argument('--batch_size', default=50,  help='Batch size for evaluation code')
    args = parser.parse_args()
    validate_arguments(args)
    net, inp_im  = choose_net(args.network)
    if args.evaluate:
        evaluate(net, args.img_list, inp_im, args.gt_labels, args.network,args.batch_size)
    else:
        predict(net, args.img_path, inp_im, args.network)

if __name__ == '__main__':
    main()
