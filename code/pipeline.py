import tensorflow as tf
import numpy as np
import skimage

#Custom
import vgg19
import vgg19_bn
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

#courtesy of https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy - not used
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


def get_layer_repr(img, layer_name="conv4_4"):
    batch1 = img.reshape(1, img.shape[0], img.shape[1], 3)
    
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        with tf.Session() as sess:

            vgg = vgg19.Vgg19(VGG_PATH + 'vgg19.npy')
            #vgg = vgg19_bn.Vgg19_bn(VGG_PATH + 'vgg19_bn.npy')
            
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
            conv = sess.run(layer, feed_dict=feed_dict)     
            
            sess.close();
            
        return conv;
        

def get_maps(img_content, img_context, K, layer_name="conv4_4"):
    
    batch1 = img_content.reshape(1, img_content.shape[0], img_content.shape[1], 3)
    batch2 = img_context.reshape(1, img_context.shape[0], img_context.shape[1], 3)

    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        with tf.Session() as sess:

            vgg = vgg19.Vgg19(VGG_PATH + 'vgg19.npy')
            images = tf.placeholder("float", [1, None, None, 3])
            
            #vgg = vgg19_bn.Vgg19_bn(VGG_PATH + 'vgg19_bn.npy')
            #images = tf.placeholder("float", [1, None, None, 3])
            vgg.build(images)
            
            writer = tf.summary.FileWriter("output", sess.graph)
            
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
            
            '''
            print(sess.run(vgg.relu6, feed_dict=feed_dict))
            prob = sess.run(vgg.prob, feed_dict=feed_dict)
            print_prob(prob[0], "../data/vgg/vgg_classes.txt")
            prob2 = sess.run(vgg.prob, feed_dict=feed_dict2)
            print_prob(prob2[0], "../data/vgg/vgg_classes.txt")
            '''
            
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
            Ws, Hs = values[0], values[1]
            
    return Ws, Hs


def show_maps(image1_path, image2_path, K, up_width=0, layer_name="conv3_4", normalize_per_k=False, cmap="Blues"):

    if up_width == 0:
        img_content = load_image2(DATA_PATH + image1_path)
        img_context = load_image2(DATA_PATH + image2_path)
    else:
        img_content = load_image2(DATA_PATH + image1_path, width=up_width)
        img_context = load_image2(DATA_PATH + image2_path, width=up_width)
    
    dicto = get_maps(img_content, img_context, K, layer_name=layer_name)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img_content)

    plt.subplot(2, 2, 2)
    plt.imshow(img_context)
    
    plt.gcf().text(0.05, 0.7, "Original ", fontsize=25)
    
    for key, values in dicto.items():
        
        if (normalize_per_k):
            Ws, Hs = normalized_k(values[0], values[1])
            Ws = np.array(Ws)
            Hs = np.array(Hs)
            img_content = skimage.transform.resize(img_content, (Ws[0].shape), mode='symmetric')
            
        else:
            #Ws, Hs = normalized_all(values[0], values[1])
            Ws = values[0]
            Hs = values[1]
            img_content = skimage.transform.resize(img_content, (Ws[:,:,0].shape), mode='symmetric')           
        
        
        img_context = skimage.transform.resize(img_context, (Hs[0].shape), mode='symmetric')
            
        for i in range(0, K):

            plt.figure(figsize=(15, 10))
            plt.gcf().text(0.05, 0.7, "k" + str(i+1) + "/" + str(K), fontsize=30)
            plt.gcf().text(0.885, 0.7, key.replace('conv',''), fontsize=30)

            plt.subplot(2, 2, 1)
            
            if (normalize_per_k):
                plt.imshow(Ws[i], cmap=cmap, vmin=0, vmax=1)
            else:
                plt.imshow(Ws[:,:,i], cmap=cmap, vmin=0, vmax=1)
                
            plt.imshow(img_content, vmin=0, vmax=1, alpha=0.2)
                
            plt.subplot(2, 2, 2)
            plt.imshow(Hs[i], cmap=cmap, vmin=0, vmax=1)
            plt.imshow(img_context, vmin=0, vmax=1, alpha=0.2)

            plt.show()



    