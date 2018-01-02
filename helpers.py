from __future__ import division

# Utils
import skimage
import skimage.io
import skimage.transform
import numpy as np
from sklearn import decomposition
from matplotlib import pyplot as plt

from collections import OrderedDict

import tensorflow as tf
import vgg19

# DCRF
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import scipy
import scipy.ndimage


import argparse
import os
import scipy.misc as spm
import scipy.ndimage as spi
import scipy.sparse as sps
import numpy as np
import tensorflow as tf

from sklearn.utils.extmath import softmax

VGG_PATH = 'data/vgg/'
DATA_PATH = 'data/'

'''
====================
        I O
====================
'''
# returns image of shape [224, 224, 3]
def load_image(image_path):
    
    # load image
    img = skimage.io.imread(image_path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    
    # we crop image to get a square from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224), mode='symmetric')
    
    return resized_img

# Credits https://github.com/machrisaa/tensorflow-vgg
def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
        
    return skimage.transform.resize(img, (int(ny), int(nx)), mode='constant')

'''
====================
        V G G
====================
'''


# Expects a built VGG model from vgg19
def eval_layers(sess, image, vgg, layer_names=["conv4_4"]):
    assert layer_names

    batch = image[None, ...]

    layer_dict = {
        'conv1_1': vgg.conv1_1,
        'conv1_2': vgg.conv1_2,
        'conv2_1': vgg.conv2_1,
        'conv2_2': vgg.conv2_2,
        'conv3_1': vgg.conv3_1,
        'conv3_2': vgg.conv3_2,
        'conv3_3': vgg.conv3_3,
        'conv3_4': vgg.conv3_4,
        'conv4_1': vgg.conv4_1,
        'conv4_2': vgg.conv4_2,
        'conv4_3': vgg.conv4_3,
        'conv4_4': vgg.conv4_4,
        'conv5_1': vgg.conv5_1,
        'conv5_2': vgg.conv5_2,
        'conv5_3': vgg.conv5_3,
        'conv5_4': vgg.conv5_4,
        'hypercolumn': vgg.hypercolumn,
    }

    # check that each strings match the dict
    check = [layer_dict.get(layer_name) is not None for layer_name in layer_names]
    assert(all(x is not None for x in check))

    layers = {layer_name: sess.run(layer_dict.get(layer_name), {"images:0": batch}) for layer_name in layer_names}
    return OrderedDict(sorted(layers.items()))


def get_layers(vgg, layer_names=["conv4_4"]):
    assert layer_names

    layer_dict = {
        'conv1_1': vgg.conv1_1,
        'conv1_2': vgg.conv1_2,
        'conv2_1': vgg.conv2_1,
        'conv2_2': vgg.conv2_2,
        'conv3_1': vgg.conv3_1,
        'conv3_2': vgg.conv3_2,
        'conv3_3': vgg.conv3_3,
        'conv3_4': vgg.conv3_4,
        'conv4_1': vgg.conv4_1,
        'conv4_2': vgg.conv4_2,
        'conv4_3': vgg.conv4_3,
        'conv4_4': vgg.conv4_4,
        'conv5_1': vgg.conv5_1,
        'conv5_2': vgg.conv5_2,
        'conv5_3': vgg.conv5_3,
        'conv5_4': vgg.conv5_4,
        'hypercolumn': vgg.hypercolumn,
    }

    # check that each strings match the dict
    check = [layer_dict.get(layer_name) is not None for layer_name in layer_names]
    assert(all(x is not None for x in check))

    layers = {layer_name: layer_dict.get(layer_name) for layer_name in layer_names}
    return OrderedDict(sorted(layers.items()))


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


'''
====================
    Masks
====================
'''


# Maps shape should be [1, H, W, C]
def compute_affinity_matrix(content_maps, reference_maps):

    content_maps = np.squeeze(content_maps, axis=0)
    reference_maps = np.squeeze(reference_maps, axis=0)

    assert(content_maps.shape[2] == reference_maps.transpose().shape[0])

    content_maps = content_maps.reshape(content_maps.shape[0] * content_maps.shape[1], content_maps.shape[2])
    reference_maps = reference_maps.reshape(reference_maps.shape[0] * reference_maps.shape[1], reference_maps.shape[2])

    return content_maps.dot(reference_maps.transpose())


# Assumes affinity matrix from @compute_affinity_matrix, shape_r/shape_l = [H,W] from layer
def get_masks(K, affinity_matrix, shape_r, shape_l, orphan=False, normalize=False, soft_temp=0.1):
    print("Computing {} masks".format(K))

    estimator = decomposition.NMF(n_components=K, init='random', tol=5e-3, random_state=0)
    L = estimator.fit_transform(affinity_matrix)
    R = estimator.components_

    L = L.transpose().reshape(K, shape_r[0], shape_r[1])
    R = R.reshape(K, shape_l[0], shape_l[1])


    if normalize:
        '''
        maxLR = max(L.max(), R.max())
        print(maxLR)
        L /= maxLR;
        R /= maxLR;
        '''
        for k in range(K):
            maxL = L[k].max()
            L[k] /= maxL

            maxR = R[k].max()
            R[k] /= maxR;

    '''else:
        for k in range(K):
            maxLR = max(L[k].max(), R[k].max())
            print(maxLR)
            L[k] /= maxLR;
            R[k] /= maxLR;
            '''

    if orphan:
        '''L_all = L.sum(axis=0)
        # np.clip(L_all, 0.0, 1.0, L_all)
        L_all /= L_all.max()
        L_orphan = 1 - L_all
        L = np.concatenate((L, L_orphan[None, ...]))

        R_all = R.sum(axis=0)
        # np.clip(R_all, 0.0, 1.0, R_all)
        R_all /= R_all.max()
        R_orphan = 1 - R_all
        R = np.concatenate((R, R_orphan[None, ...]))
        '''

        L_orphan = np.ones(L[0].shape) * L.mean()
        R_orphan = np.ones(R[0].shape) * R.mean()

        L = np.concatenate((L, L_orphan[None, ...]))
        R = np.concatenate((R, R_orphan[None, ...]))

        Lsm = softmax(L.reshape((L.shape[0], -1)).transpose() / soft_temp).transpose().reshape(L.shape)
        Rsm = softmax(R.reshape((R.shape[0], -1)).transpose() / soft_temp).transpose().reshape(R.shape)

    return Lsm, Rsm


def show_masks(original_left, original_right, L, R, K, cmap="Blues", normalized=True, show_original=True, show_axis=False, vmax=1):

    plt.figure(figsize=(10, 8))
    plt.title("Original")

    plt.subplot(2, 2, 1)
    plt.imshow(original_left)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(original_right)
    plt.axis('off')
    img_content = skimage.transform.resize(original_left, (L[0].shape), mode='symmetric')
    img_context = skimage.transform.resize(original_right, (R[0].shape), mode='symmetric')

    for k in range(K):

        plt.figure(figsize=(10, 8))
        #plt.gcf().text(0.05, 0.7, "k" + str(i + 1) + "/" + str(K), fontsize=30)
        #plt.gcf().text(0.885, 0.7, key.replace('conv', ''), fontsize=30)

        plt.title("k = {}".format(k))

        plt.subplot(2, 2, 1)
        if normalized:
            plt.imshow(L[k], cmap=cmap, vmin=0, vmax=vmax)
            if not show_axis:
                plt.axis('off')

            #plt.colorbar()
            if show_original:
                plt.imshow(img_content, vmin=0, vmax=vmax, alpha=0.2)

            plt.subplot(2, 2, 2)
            plt.imshow(R[k], cmap=cmap, vmin=0, vmax=vmax)
            #plt.colorbar()
            if show_original:
                plt.imshow(img_context, vmin=0, vmax=vmax, alpha=0.2)

            if not show_axis:
                plt.axis('off')

        else:
            plt.imshow(L[k], cmap=cmap)
            #plt.colorbar()
            if show_original:
                plt.imshow(img_content, alpha=0.2)
            if not show_axis:
                plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.imshow(R[k], cmap=cmap)
            #plt.colorbar()
            if show_original:
                plt.imshow(img_context, alpha=0.2)
            plt.colormaps()

            if not show_axis:
                plt.axis('off')

        plt.show()


'''
====================
    DCRF
====================
'''

def refine_mask(mask_in=None):
    mask_in = mask_in.astype(np.uint8)
    mask_out = scipy.ndimage.morphology.binary_fill_holes(mask_in)  # fill in the holes
    return mask_out


crf_params = {}
crf_params['sxy'] = (50, 50)
crf_params['srgb'] = (3, 3, 3)
crf_params['compat'] = 5
crf_params['d_infer'] = 5


def dcrf_refine(img, mask):

    img_sz = img.shape[0:2]
    if img_sz[0] != mask.shape[0]:
        mask = skimage.transform.resize(mask, img_sz, mode='constant', order=1)

    ## refine with CRF
    labels_c = mask[None]  # labels.reshape((1,img_sz[0],img_sz[1]))
    labels_c = np.concatenate([1 - labels_c, labels_c], axis=0)
    d = dcrf.DenseCRF2D(img_sz[1], img_sz[0], 2)
    U = unary_from_softmax(labels_c, scale=0.6)
    d.setUnaryEnergy(U)

    d.addPairwiseBilateral(sxy=crf_params['sxy'], srgb=crf_params['srgb'], rgbim=img.astype(np.uint8),
                           compat=crf_params['compat'],
                           kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(crf_params['d_infer'])
    Q = np.array(Q)[0]
    labels_c = refine_mask(Q.reshape(img_sz) < 0.5)
    return labels_c



'''
========================================
    MATTING LAPLACIAN CODE FROM 
========================================
'''

def getLaplacian(img):
    h, w, _ = img.shape
    coo = getlaplacian1(img, np.zeros(shape=(h, w)), 1e-5, 1).tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def getlaplacian1(i_arr, consts, epsilon=1e-5, win_rad=1):
    neb_size = (win_rad * 2 + 1) ** 2
    h, w, c = i_arr.shape
    img_size = w * h
    consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_rad * 2 + 1, win_rad * 2 + 1)))

    indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
    tlen = int((-consts[win_rad:-win_rad, win_rad:-win_rad] + 1).sum() * (neb_size ** 2))
    row_inds = np.zeros(tlen)
    col_inds = np.zeros(tlen)
    vals = np.zeros(tlen)
    l = 0
    for j in range(win_rad, w - win_rad):
        for i in range(win_rad, h - win_rad):
            if consts[i, j]:
                continue
            win_inds = indsM[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1]
            win_inds = win_inds.ravel(order='F')
            win_i = i_arr[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1, :]
            win_i = win_i.reshape((neb_size, c), order='F')
            win_mu = np.mean(win_i, axis=0).reshape(c, 1)
            win_var = np.linalg.inv(
                np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu, win_mu.T) + epsilon / neb_size * np.identity(
                    c))

            win_i2 = win_i - np.repeat(win_mu.transpose(), neb_size, 0)
            tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size

            ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
            row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
            col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
            vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
            l += neb_size ** 2

    vals = vals.ravel(order='F')[0: l]
    row_inds = row_inds.ravel(order='F')[0: l]
    col_inds = col_inds.ravel(order='F')[0: l]
    a_sparse = sps.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))

    sum_a = a_sparse.sum(axis=1).T.tolist()[0]
    a_sparse = sps.diags([sum_a], [0], shape=(img_size, img_size)) - a_sparse

    return a_sparse


