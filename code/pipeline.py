import tensorflow as tf
import numpy as np

#Custom
import vgg19
from helpers import load_image, print_prob, load_image2

#IO
import matplotlib.pyplot as plt

#Datastructure
from collections import OrderedDict

VGG_PATH = '../data/vgg/'
DATA_PATH = '../data/'

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

#courtesy of https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def normalized_all(a, b):
    
    min_ab = np.min(a) if np.min(a) <= np.min(b) else np.min(b)
    a2 = a - min_ab
    b2 = b - min_ab
    max_ab = np.max(a2) if np.max(a2) >= np.max(b2) else np.max(b2)
    
    return a2 / max_ab, b2 / max_ab

def normalized_k(a, b):
    
    a2, b2 = [],[]
    
    for i in range(0, a.shape[2]):
        ret = normalized_all(a[:,:,i], b[i])
        a2.append(ret[0])
        b2.append(ret[1])
        
    return a2,b2

    
def get_maps(img_content, img_context, K, layer_name="conv4_4"):
    
    batch1 = img_content.reshape(1, img_content.shape[0], img_content.shape[1], 3)
    batch2 = img_context.reshape(1, img_context.shape[0], img_context.shape[1], 3)

    tf.reset_default_graph()
    with tf.device('/gpu:0'):
        with tf.Session() as sess:

            vgg = vgg19.Vgg19(VGG_PATH + 'vgg19.npy')
            images = tf.placeholder("float", [1, None, None, 3])
            vgg.build(images)
            
            layer_dict = {
               
                'conv2_1' : vgg.conv2_1,
                'conv2_2' : vgg.conv2_2,

                'conv3_1' : vgg.conv3_1,
                'conv3_2' : vgg.conv3_2,   
                'conv3_3' : vgg.conv3_3,
                'conv3_4' : vgg.conv3_4,  

                'conv4_1' : vgg.conv4_1,
                'conv4_2' : vgg.conv4_2,
                'conv4_3' : vgg.conv4_3,
                'conv4_4' : vgg.conv4_4, 

                'conv5_1' : vgg.conv5_1,
                'conv5_2' : vgg.conv5_2,
                'conv5_3' : vgg.conv5_3,
                'conv5_4' : vgg.conv5_4, 
            }
            
            layer = layer_dict.get(layer_name)

            
            if (layer == None):
                raise ValueError("layer specified not found in dictionnary")
            
            feed_dict = {images: batch1}
            feed_dict2 = {images: batch2}
            conv = [sess.run(layer, feed_dict=feed_dict)[0], sess.run(layer, feed_dict=feed_dict2)[0]]     

            sess.close();
        
  
    Ws, Hs = get_K_repr(K, conv)
       
    repr_dict = {
        layer_name : [Ws, Hs],   
    }

    return OrderedDict(sorted(repr_dict.items()))
    

def gen_maps(image1_path, image2_path, K, up_width=0, layer_name="conv3_4", normalize_per_k=True):
    
    if up_width == 0:
        img_content = load_image2(DATA_PATH + image1_path)
        img_context = load_image2(DATA_PATH + image2_path)
    else:
        img_content = load_image2(DATA_PATH + image1_path, width=up_width)
        img_context = load_image2(DATA_PATH + image2_path, width=up_width)
    
    dicto = get_maps(img_content, img_context, K, layer_name=layer_name)

    
    for key, values in dicto.items():
        
        if (normalize_per_k):
            Ws, Hs = normalized_k(values[0], values[1])
            Ws = np.array(Ws)
            Hs = np.array(Hs)
        else:
            Ws, Hs = normalized_all(values[0], values[1])
            
    return Ws, Hs

    