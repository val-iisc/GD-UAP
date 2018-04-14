# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

monodepth_parameters = namedtuple('parameters',
                                  'encoder, '
                                  'height, width, '
                                  'batch_size')


class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, params, input_tf, reuse_variables=None, model_index=0):
        self.params = params
        # As we are doing only one pass
        self.left = input_tf
        self.right = None
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        self.build_model(input_tf)
        self.build_outputs()

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(
            p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:, 3:-1, 3:-1, :]

    def build_vgg(self):
        # set convenience functions
        conv = self.conv
        upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input,  32, 7)  # H/2
            conv2 = self.conv_block(conv1,             64, 5)  # H/4
            conv3 = self.conv_block(conv2,            128, 3)  # H/8
            conv4 = self.conv_block(conv3,            256, 3)  # H/16
            conv5 = self.conv_block(conv4,            512, 3)  # H/32
            conv6 = self.conv_block(conv5,            512, 3)  # H/64
            conv7 = self.conv_block(conv6,            512, 3)  # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7,  512, 3, 2)  # H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7 = conv(concat7,  512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6,  512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5,  256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4,  128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3,   64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2,   32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def build_resnet50(self):
        # set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2)  # H/2  -   64D
            pool1 = self.maxpool(conv1,           3)  # H/4  -   64D
            conv2 = self.resblock(pool1,      64, 3)  # H/8  -  256D
            conv3 = self.resblock(conv2,     128, 4)  # H/16 -  512D
            conv4 = self.resblock(conv3,     256, 6)  # H/32 - 1024D
            conv5 = self.resblock(conv4,     512, 3)  # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5,   512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4,   128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3,    64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2,    32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def build_model(self, input_tf):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.left_pyramid = self.scale_pyramid(self.left,  4)

                self.model_input = self.left  # +noise

                # build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                else:
                    return None

    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est = [tf.expand_dims(
                d[:, :, :, 0], 3) for d in self.disp_est]
            print(len(self.disp_left_est))
            print(self.disp1.shape, self.disp2.shape)
            # print(self.disp_left_est.shape)
            self.disp_right_est = [tf.expand_dims(
                d[:, :, :, 1], 3) for d in self.disp_est]
