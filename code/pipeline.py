import tensorflow as tf
import numpy as np

#Custom
import vgg19
from helpers import load_image, print_prob

#IO
import matplotlib.pyplot as plt

#Datastructure
from collections import OrderedDict

VGG_PATH = '../data/vgg/'

from sklearn import decomposition  

# Assumes layer is a [2, W, H, C] array
def get_K_repr(K, layer):

    content_maps = layer[0]
    reference_maps = layer[1] 

    content_maps = content_maps.reshape(content_maps.shape[0]*content_maps.shape[1], content_maps.shape[2])
    reference_maps = reference_maps.reshape(reference_maps.shape[0]*reference_maps.shape[1], reference_maps.shape[2]).transpose()
    
    res = content_maps.dot(reference_maps)

    estimator = decomposition.NMF(n_components = K, init = 'random', tol=5e-3, random_state=0)    
    W = estimator.fit_transform(res)
    H = estimator.components_
    
    W = W.reshape(layer[0].shape[0], layer[0].shape[1], K)
    H = H.reshape(K, layer[1].shape[0], layer[1].shape[1])
    
    return W, H


def get_maps(img_content, img_context, K):
    
    content_shape = img_content.shape
    context_shape = img_context.shape
    
    #print(content_shape)
    #print(context_shape)
    
    batch = np.concatenate((img_content.reshape((1, content_shape[0], content_shape[1], 3)), img_context.reshape((1, context_shape[0], context_shape[1], 3))), 0)

    tf.reset_default_graph()
    with tf.device('/gpu:0'):
        with tf.Session() as sess:

            images = tf.placeholder("float", [2, 224, 224, 3])
            feed_dict = {images: batch}

            vgg = vgg19.Vgg19(VGG_PATH + 'vgg19.npy')

            vgg.build(images)

            '''
            conv2_1 = sess.run(vgg.conv2_1, feed_dict=feed_dict)
            conv2_2 = sess.run(vgg.conv2_2, feed_dict=feed_dict)

            conv3_1 = sess.run(vgg.conv3_1, feed_dict=feed_dict)
            conv3_2 = sess.run(vgg.conv3_2, feed_dict=feed_dict)

            conv4_1 = sess.run(vgg.conv4_1, feed_dict=feed_dict)
            conv4_2 = sess.run(vgg.conv4_2, feed_dict=feed_dict)
            conv4_3 = sess.run(vgg.conv4_3, feed_dict=feed_dict)
            

            conv5_1 = sess.run(vgg.conv5_1, feed_dict=feed_dict)
            conv5_2 = sess.run(vgg.conv5_2, feed_dict=feed_dict)
            conv5_3 = sess.run(vgg.conv5_3, feed_dict=feed_dict)
            conv5_4 = sess.run(vgg.conv5_4, feed_dict=feed_dict)
            
            
            conv3_3 = sess.run(vgg.conv3_3, feed_dict=feed_dict)
           
            conv3_4 = sess.run(vgg.conv3_4, feed_dict=feed_dict)  
            '''
            
            conv4_4 = sess.run(vgg.conv4_4, feed_dict=feed_dict)
            
            sess.close();
        
    '''
    Ws_21, Hs_21 = get_K_repr(K, conv2_1)
    Ws_22, Hs_22 = get_K_repr(K, conv2_2)   
    
    
    Ws_31, Hs_31 = get_K_repr(K, conv3_1)
    Ws_32, Hs_32 = get_K_repr(K, conv3_2)

         
    Ws_41, Hs_41 = get_K_repr(K, conv4_1)
    Ws_42, Hs_42 = get_K_repr(K, conv4_2)

    
    Ws_51, Hs_51 = get_K_repr(K, conv5_1)
    Ws_52, Hs_52 = get_K_repr(K, conv5_2)
    Ws_53, Hs_53 = get_K_repr(K, conv5_3)
    Ws_54, Hs_54 = get_K_repr(K, conv5_4)
    
    
    Ws_43, Hs_43 = get_K_repr(K, conv4_3)
   
    Ws_33, Hs_33 = get_K_repr(K, conv3_3)
    
    Ws_34, Hs_34 = get_K_repr(K, conv3_4)
    '''
    Ws_44, Hs_44 = get_K_repr(K, conv4_4)
    
    '''
        'conv2_1' : [Ws_21, Hs_21],
        'conv2_2' : [Ws_22, Hs_22],
        
        'conv3_1' : [Ws_31, Hs_31],
        'conv3_2' : [Ws_32, Hs_32],   
        'conv3_3' : [Ws_33, Hs_33],
        'conv3_4' : [Ws_34, Hs_34],  
    
        'conv4_1' : [Ws_41, Hs_41],
        'conv4_2' : [Ws_42, Hs_42],
        'conv4_3' : [Ws_43, Hs_43],
        
         
        'conv5_1' : [Ws_51, Hs_51],
        'conv5_2' : [Ws_52, Hs_52],
        'conv5_3' : [Ws_53, Hs_53],
        'conv5_4' : [Ws_54, Hs_54], 
        
    '''
       
    repr_dict = {
 
        'conv4_4' : [Ws_44, Hs_44], 
    }

    return OrderedDict(sorted(repr_dict.items()))

        
    #prob = sess.run(vgg.prob, feed_dict=feed_dict)
    #print_prob(prob[0], VGG_PATH + 'vgg_classes.txt')
    #print_prob(prob[1], VGG_PATH + 'vgg_classes.txt')
    
    
#courtesy of https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
    
def get_maps2(img_content, img_context, K):
    
    content_shape = img_content.shape
    context_shape = img_context.shape
    
    batch1 = img_content.reshape(1, content_shape[0], content_shape[1], 3)
    batch2 = img_context.reshape(1, context_shape[0], context_shape[1], 3)

    tf.reset_default_graph()
    with tf.device('/gpu:0'):
        with tf.Session() as sess:

            vgg = vgg19.Vgg19(VGG_PATH + 'vgg19.npy')
            images = tf.placeholder("float", [1, None, None, 3])
            vgg.build(images)
            
            feed_dict = {images: batch1}
            feed_dict2 = {images: batch2}
            conv4_4 = [sess.run(vgg.conv4_4, feed_dict=feed_dict)[0], sess.run(vgg.conv4_4, feed_dict=feed_dict2)[0]]     

            #conv3_4 = sess.run(vgg.conv3_4, feed_dict=feed_dict)
            sess.close();
        
  
    Ws, Hs = get_K_repr(K, conv4_4)
    #[array_s3[0], array_s3_2[0]]
       
    repr_dict = {
        'conv4_4' : [normalized(Ws), normalized(Hs)],   
    }

    return OrderedDict(sorted(repr_dict.items()))
    
