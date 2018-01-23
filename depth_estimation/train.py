
import sys
sys.insert(0,'monodepth_files/')
from utils.functions import *
from monodepth_model import * 
import tensorflow as tf
import numpy as np
import argparse
import os
import time
import math
import utils.losses as losses
#import pickle

def get_net(params,checkpoint_file,batch_size):
    size = [256,512]
    input_image = tf.placeholder(shape=[batch_size, size[0], size[1], 3],dtype='float32', name='input_image')
    # initializing adversarial image
    adv_image = tf.Variable(tf.random_uniform([1,size[0],size[1],3],minval=-10/256.0,maxval=10/256.0), name='noise_image', dtype='float32')
    # clipping for imperceptibility constraint
    adv_image = tf.clip_by_value(adv_image,-10/256.0,10/256.0)
    input_batch = tf.add(input_image,adv_image)
    
    model = MonodepthModel(params,input_batch)
    
    # get the layers for loading the pretrained weights
    net_varlist = [v for v in tf.get_collection(tf.GraphKeys.VARIABLES) if v.name not in ['noise_image:0']]

    saver = tf.train.Saver(var_list =net_varlist)
    print(checkpoint_file)
    def restore_func(sess,checkpoint_path=checkpoint_file,saver=saver):
      saver.restore(sess,checkpoint_path)
    
    return model,input_image, adv_image,restore_func

def get_optim_layers():
    
    optim_layers = []
    operations  = tf.get_default_graph().get_operations()
    for op in operations:
        if 'encoder' in op.name and op.type == u'Elu':
            optim_layers.append(op.outputs)
        if 'decoder' in op.name and op.type == u'Elu':
            optim_layers.append(op.outputs)
            
    print("The optimization layers",optim_layers)
    return optim_layers

def rescale_checker_function(check,sat,sat_change,sat_min):
    value = (sat_change<check and sat>sat_min)
    return value


def get_update_operation_func(train_type,in_im,sess,update,batch_size,size,img_list):
    if train_type == 'no_data':
        def updater(noiser,sess=sess,update=update):
            sess.run(update,feed_dict = {in_im: noiser})
    elif train_type =='with_noise':
        def updater(noiser,sess=sess,update=update,in_im=in_im,batch_size = batch_size,size=size):
            image_i = 'misc/gaussian_noise.png'
            for j in range(batch_size):
                noiser[j:j+1] = np.copy(utils.img_preprocess_depth(image_i,size=size,augment=True))
            sess.run(update,feed_dict={in_im:noiser})
    elif train_type =='with_data':
        def updater(noiser,sess=sess,update=update,in_im=in_im,batch_size = batch_size,size=size,img_list=img_list):
            rander = np.random.randint(low=0,high=(len(img_list)-batch_size-1))
            for j in range(batch_size):
                noiser[j:j+1] = np.copy(utils.img_preprocess_depth(img_list[rander+j].strip(),size=size,augment=True))
            #print(noiser.shape)
            sess.run(update,feed_dict={in_im:noiser})
    return updater

def train(net, in_im, ad_im, opt_layers,
net_name,train_type,rescale_type,check_val,lamb_val,img_list_file=None,restore_func=None,batch_size=1):
    
    ###Vanilla Version
    cost = -losses.l2_outputs(opt_layers)
    tvars = tf.trainable_variables()[0]
    print(tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1/256.0)
    grads = optimizer.compute_gradients(cost,tvars)
    update = optimizer.apply_gradients(grads)
    
    data_path =os.path.join('data','preprocess_depth_small.npy')
    size = [256,512]
    imgs = np.load(data_path)#[:200,:,:,:]
    print('Loaded mini Validation Set')
    
    ## constants
    loss_val = np.Inf # current fooling rate
    max_iter = 40000
    stopping = 0 # early stopping condition
    t_s = time.time()
    ### New constants
    check = 0.00001
    prev_check = 0
    rescaled = False
    stop_check= False
    #batch_size = 32
    noiser = np.zeros((batch_size,size[0],size[1],3))#np.random.uniform(high=123.0,low=-123.0,size = (1000,224,224,3))
    rescaled = False
    if train_type == 'with_data':
        img_list = open(img_list_file).readlines()#[:5000]
    else:
        img_list = None

    print "Starting {:} training...".format(net_name)
    

    ### Saturation Measure
    saturation = tf.div(tf.reduce_sum(tf.to_float(tf.equal(tf.abs(ad_im),10/256.0))),tf.to_float(tf.size(ad_im)))
    # rate of change of percentage change
    sat_prev = 0
    sat = 0
    sat_change = 0
    sat_min=0.5
    
    ## rescaler
    assign_op = tvars.assign(tf.divide(tvars,2.0))
    #swapper = tf.placeholder('float',[1,256,512,3])
    #swap_op = tvars.assign(swapper)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        update_op = get_update_operation_func(train_type,in_im,sess,update,batch_size,size,img_list)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        restore_func(sess)
        for i in range(max_iter):
            update_op(noiser)
            # calculate variables
            sat_prev = np.copy(sat)
            sat = sess.run(saturation)
            sat_change = abs(sat-sat_prev)
            check_dif = i -prev_check
            if i%100==0:
                print('iter',i,'current_saturation',sat,'sat_change',sat_change)

            # check for saturation
            if rescale_checker_function(check,sat,sat_change,sat_min):
                 rescaled =True
            # validation time
            if not stop_check and ((check_dif>200 and rescaled == True) or check_dif==400):
                print('checking performance')
                iters = int(math.ceil(imgs.shape[0]/float(batch_size)))
                temp = 0
                prev_check = i
    	        for j in range(iters-1):
    	            l = j*batch_size
    	            L = min((j+1)*batch_size,999)
                    cur_cost  = sess.run(cost, feed_dict={in_im: imgs[l:L]})
                    temp += cur_cost
                current_rate = temp/img.shape[0]
                print('current_loss', current_rate,'current_iter',i)
                if current_rate<=loss_val:
                    print('best_performance_till_now')
                    stopping =0
                    loss_val = current_rate
                    im = sess.run(ad_im)
                    name = 'perturbations/'+net_name+'_'+train_type+'.npy'
                    np.save(name,im)
                else:
                    stopping+=1
                if stopping==25:
                    # As the loss on validation set might not be discriminative enough, 
                    # we also try the last perturbation.
                    print('Val best out')
                    im = sess.run(ad_im)
                    name = 'perturbations/last_'+net_name+'_'+train_type+'.npy'
                    np.save(name,im)
                    stop_check =True
                    break
            
            if rescale_checker_function(check,sat,sat_change,sat_min):
                sess.run(assign_op)
                print('reached_saturation',sat,sat_change,'criteria',check,'iter',i)
                rescaled = False
                prev_check = i
        print('training_done', time.time()-t_s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior_type', default='googlenet', help='The network eg. googlenet')
    parser.add_argument('--img_list', default='None', help='The network eg. googlenet')
    parser.add_argument('--checkpoint_file', default='None', help='The network eg. googlenet')
    parser.add_argument('--batch_size', default='None', help='The network eg. googlenet')
    parser.add_argument('--encoder', default='None', help='The network eg. googlenet')
    args = parser.parse_args()
    if args.img_list == 'None':
        args.img_list = None
    params = monodepth_parameters(encoder=args.encoder,height=256,width=512,batch_size=args.batch_size)
    net, inp_im, ad_im, restore_func  = get_net(params,args.checkpoint_file,int(float(args.batch_size)))
    opt_layers = get_optim_layers()#get_optim_layers()
    train(net, inp_im, ad_im, opt_layers,
    args.encoder,args.train_type,args.img_list,restore_func,int(float(args.batch_size)))

if __name__ == '__main__':
    main()
