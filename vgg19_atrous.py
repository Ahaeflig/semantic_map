import os
import tensorflow as tf

import numpy as np
import time
import inspect

#Datastructure
from collections import OrderedDict

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19_Atrous:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19_Atrous)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        #print("npy file loaded")

    def preprocess(self, rgb_scaled):
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        return bgr

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        rgb_scaled = rgb * 255.0
        bgr = self.preprocess(rgb_scaled)
        
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        rate = 1
        self.conv3_1 = self.conv_layer_atrous(self.pool2, "conv3_1", rate)
        self.conv3_2 = self.conv_layer_atrous(self.conv3_1, "conv3_2", rate)
        self.conv3_3 = self.conv_layer_atrous(self.conv3_2, "conv3_3", rate)
        self.conv3_4 = self.conv_layer_atrous(self.conv3_3, "conv3_4", rate)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        rate = 1;
        self.conv4_1 = self.conv_layer_atrous(self.pool3, "conv4_1", rate)
        self.conv4_2 = self.conv_layer_atrous(self.conv4_1, "conv4_2", rate)
        self.conv4_3 = self.conv_layer_atrous(self.conv4_2, "conv4_3", rate)
        self.conv4_4 = self.conv_layer_atrous(self.conv4_3, "conv4_4", rate)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4', stride=1)

        rate = 2;
        self.conv5_1 = self.conv_layer_atrous(self.pool4, "conv5_1", rate)
        self.conv5_2 = self.conv_layer_atrous(self.conv5_1, "conv5_2", rate)
        self.conv5_3 = self.conv_layer_atrous(self.conv5_2, "conv5_3", rate)
        self.conv5_4 = self.conv_layer_atrous(self.conv5_3, "conv5_4", rate)


        layer_dict = {
            'conv2_1' : self.conv2_1,
            'conv2_2' : self.conv2_2,
            'conv3_1' : self.conv3_1,
            'conv3_2' : self.conv3_2,   
            'conv3_3' : self.conv3_3,
            'conv3_4' : self.conv3_4,  
            'conv4_1' : self.conv4_1,
            'conv4_2' : self.conv4_2,
            'conv4_3' : self.conv4_3,
            'conv4_4' : self.conv4_4, 
            'conv5_1' : self.conv5_1,
            'conv5_2' : self.conv5_2,
            'conv5_3' : self.conv5_3,
            'conv5_4' : self.conv5_4, 
        }
        
        ordered_layer_dict = OrderedDict(sorted(layer_dict.items()))
        layers = [ordered_layer_dict.get(layer_name) for layer_name in ordered_layer_dict]
        
        resized_layer = [tf.image.resize_images(l, (32, 32)) for l in layers]
        
        #tf.image.resize_images(image, new_height_and_width[0], new_height_and_width[1])
        
        #self.hypercolumn = (tf.concat(0, [l for l in resized_layer], name="hypercolumn"))
        self.hypercolumn = 0

        #print(("build model finished: %ds" % (time.time() - start_time)))
        

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name, stride=2):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu


    def conv_layer_atrous(self, bottom, name, rate):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.atrous_conv2d(bottom, filt, rate=rate, padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")