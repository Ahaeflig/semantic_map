import os
import tensorflow as tf

import numpy as np
import time
import inspect


#VGG_MEAN = [103.939, 116.779, 123.68]
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]


BN_EPSILON = 0.001

class Vgg19_bn:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19_bn)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19_bn.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        #print("npy file loaded")
        
    def normalize_batch(self, d, mean_wanted, std_wanted):
        curr_mean, curr_var = tf.nn.moments(d, axes=[1])
        return mean_wanted + (d - curr_mean) * (std_wanted / tf.sqrt(curr_var))
        
        
    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        #print("build model started")

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        '''
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
       
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        '''
        
        rgb_scaled = tf.concat(axis=3, values=[
            self.normalize_batch(red, VGG_MEAN[0], VGG_STD[0]),
            self.normalize_batch(green, VGG_MEAN[1], VGG_STD[1]),
            self.normalize_batch(blue, VGG_MEAN[2], VGG_STD[2]),
        ])
        
        print(rgb_scaled.shape)
        
        self.conv1_1 = self.conv_layer(rgb_scaled, "conv1_1", "features.0", "features.1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2", "features.3", "features.4")
        
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1", "features.7", "features.8")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2", "features.10", "features.11")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1", "features.14", "features.15")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2", "features.17", "features.18")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3", "features.20", "features.21")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4", "features.23", "features.24")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1", "features.27", "features.28")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2", "features.30", "features.31")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3", "features.33", "features.34")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4", "features.36", "features.37")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1", "features.40", "features.41")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2", "features.43", "features.44")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3", "features.46", "features.47")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4", "features.49", "features.50")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        '''
        self.fc6 = self.fc_layer(self.pool5, "fc6")
        
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        '''
        
        self.data_dict = None
        
        #print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, feature_name, bn_name):
        with tf.variable_scope(name):
            
            filt = self.get_conv_filter(feature_name)
            
            #(64, 3, 3, 3) => pytorch
            #[filter_height, filter_width, in_channels, out_channels] => tf
            
            filt = tf.transpose(filt, [2, 3, 1, 0])
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            
            #print(filt.get_shape())
            #print(conv.get_shape())

            conv_biases = self.get_bias(feature_name)
            bias = tf.nn.bias_add(conv, conv_biases)
            
            # TODO add BN
            beta = self.get_bias(bn_name)
            gamma = self.get_weight(bn_name)
            
            mean = self.get_bn_mean(bn_name)
            variance = self.get_bn_var(bn_name)
            
            bn = tf.nn.batch_normalization(bias, mean, variance, beta, gamma, BN_EPSILON)
            #bn = tf.nn.batch_normalization(bias, mean, variance, gamma, beta, BN_EPSILON)
            
            
            relu = tf.nn.relu(bn)
            return relu

    def fc_layer(self, bottom, name, npy_name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_weight(npy_name)
            biases = self.get_bias(npy_name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
        
    def get_bn_mean(self, name):
        return tf.constant(self.data_dict[name + ".running_mean"], name="bn_mean")

    def get_bn_var(self, name):
        return tf.constant(self.data_dict[name + ".running_var"] , name="bn_var")
    
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name + ".weight"], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name + ".bias"], name="biases")

    def get_weight(self, name):
        return tf.constant(self.data_dict[name + ".weight"], name="weights")