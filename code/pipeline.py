import tensorflow as tf
import numpy as np

#Custom
import vgg19
from helpers import load_image, print_prob

#IO
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file

DATA_PATH = '../data/'
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
    
    a = layer[0].shape[0]
    
    W = W.reshape(layer[0].shape[0], layer[0].shape[1], K)
    H = H.reshape(K, layer[1].shape[0], layer[1].shape[1])
    
    return W, H


def get_maps(image1_path, image2_path, K):

    img_content = load_image(DATA_PATH + image1_path)
    img_context = load_image(DATA_PATH + image1_path)
    
    batch = np.concatenate((img_content.reshape((1, 224, 224, 3)), img_context.reshape((1, 224, 224, 3))), 0)


    with tf.Session() as sess:

        images = tf.placeholder("float", [2, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg19.Vgg19(VGG_PATH + 'vgg19.npy')

        vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        pool5 = sess.run(vgg.pool5, feed_dict=feed_dict)
        conv4_3 = sess.run(vgg.conv4_3, feed_dict=feed_dict)
        conv2_2 = sess.run(vgg.conv2_2, feed_dict=feed_dict)

        print_prob(prob[0], VGG_PATH + 'vgg_classes.txt')
        print_prob(prob[1], VGG_PATH + 'vgg_classes.txt')

        sess.close();

        
    K = 15
    Ws, Hs = get_K_repr(K, conv4_3)
    
    
    return [Ws, Hs]


        
