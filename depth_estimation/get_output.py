
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.misc import imread
from skimage.transform import resize  
from monodepth_model import *

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def test(params,adv_image):
    """Test function."""
    # The input and the perturbation

    input_tf = tf.placeholder(shape=[1,256,512,3],dtype='float32', name='input_image')
    pert_load = np.load(adv_image)
    adv_image = tf.constant(pert_load, dtype='float32')
    input_final = tf.add(input_tf,adv_image)
    input_final = tf.clip_by_value(input_final, 0, 1)
    model = MonodepthModel(params,input_final)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(ckpt_file)
    else:
        restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities    = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    images = open(args.filenames_file).readlines()
    input_np = np.zeros((1,256,512,3))
    for step in range(num_test_samples):
        im_path = args.data_path +images[step].split(' ')[0].strip()
        image = imread(im_path)
        image=resize(image,[256,512],mode='constant',preserve_range=True)/256.0
        input_np[0] = np.copy(image)
        #input_np[1] = np.copy(image[:,::-1,:])
        disp = sess.run(model.disp_left_est[0],feed_dict={input_tf:input_np})
        #print(disp.shape)
        disparities[step] = disp[0].squeeze()

    print('done.')

    print('writing disparities.')
    output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy',    disparities)

    print('done.')

def main():


    parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

    parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=256)
    parser.add_argument('--input_width',               type=int,   help='input width', default=512)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
    parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='output/')
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
    parser.add_argument('--perturbation',           type=str,   help='path to a specific checkpoint to load', default='')

    args = parser.parse_args()
    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size)
    adv_image = args.perturbation

    test(params,adv_image)

if __name__ == '__main__':
    main()
