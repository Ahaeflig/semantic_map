import tensorflow as tf
import numpy as np
import skimage
import skimage.transform
from helpers import *

from helpers import getLaplacian
from VOChelpers import *

# IO
import matplotlib.pyplot as plt

from smooth_local_affine import smooth_local_affine

# Datastructure
from collections import OrderedDict

# Custom
import vgg19
import vgg19_atrous

VGG_PATH = 'data/vgg/'
DATA_PATH = 'data/'

class StyleTransfer:
    #TODO could create a mask object instead of all this stuff in params
    def __init__(self, content_layer_name, style_layers_name, mask_layer_name, init_image, content_image, style_image, session, num_iter,
                 content_loss_weight, style_loss_weight, K=15, normalize=False, debug=False, orphan=True, use_dcrf=False, matting_loss=1000, tv_loss=1000, soft_temp=0.1, voc_names=[], k_style=[]):

        # Check same image size (can extend to multiple image sizes)
        assert (content_image.shape == style_image.shape)

        ''' 
        =======================================
        Store Params
        =======================================
        '''
        self.K = K

        self.content_layer_name = content_layer_name
        self.style_layers_name = style_layers_name
        self.mask_layer_name = mask_layer_name

        self.init_image = init_image
        self.content_image = content_image
        self.style_image = style_image
        self.num_iter = num_iter

        self.sess = session
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight

        self.matting_loss = matting_loss
        self.tv_loss = tv_loss

        self.debug = debug

        self.image_final = 0
        self.voc_names = voc_names

        # choose K used for style transfer
        if not k_style:
            self.k_style = np.array(range(K))
        else:
            self.k_style = k_style

        ''' 
        ==============================
        Pre-compute VGG stuff
        ==============================
        '''
        # Vgg
        # Only one graph is needed cause we assume the shape of all the images are the same
        self.vgg = vgg19.Vgg19(VGG_PATH + 'vgg19.npy')
        self.vgg_shape = [1, content_image.shape[0], content_image.shape[1], content_image.shape[2]]

        images = tf.placeholder("float", self.vgg_shape, name="images")
        self.vgg.build(images)

        # Content from content image and style image (remember eval_layers() return dicts)
        self.content_layer = eval_layers(self.sess, self.content_image, self.vgg, self.content_layer_name)
        self.style_layer = eval_layers(self.sess, self.style_image, self.vgg, self.content_layer_name)

        # Mask layers,  note that eval_layers expects array of names
        self.mask_content_layer = eval_layers(self.sess, content_image, self.vgg, [self.mask_layer_name])
        self.mask_style_layer = eval_layers(self.sess, style_image, self.vgg, [self.mask_layer_name])


        ''' VGG atrous stuff
        self.vgg_atrous = vgg19_atrous.Vgg19_Atrous(VGG_PATH + 'vgg19.npy')
        self.vgg_atrous.build(images)
        
        self.mask_content_layer = eval_layers(self.sess, content_image, self.vgg_atrous, [self.mask_layer_name])
        self.mask_style_layer = eval_layers(self.sess, style_image, self.vgg_atrous, [self.mask_layer_name])
        '''

        # Style from style image, precomputed since the gram doesn't change
        self.style_layers = eval_layers(self.sess, style_image, self.vgg, self.style_layers_name)
        self.grams_style = {layer_name: self.compute_gram(layer) for layer_name, layer in self.style_layers.items()}


        ''' 
        ==================
        Mask processing
        ==================
        '''

        # Masks between the content layer of content image and style image
        self.A = compute_affinity_matrix(self.mask_content_layer.get(self.mask_layer_name),
                                         self.mask_style_layer.get(self.mask_layer_name))

        self.orphan = 0

        # If you want to use VOC segmentation it's possible using the voc_names with voc imagenames
        if not voc_names:
            if self.K > 0:
                L, R = get_masks(K, self.A, self.mask_content_layer.get(self.mask_layer_name).shape[1:3],
                                 self.mask_style_layer.get(self.mask_layer_name).shape[1:3], orphan=orphan, normalize=normalize, soft_temp=soft_temp)

                #Add orphan masks
                if orphan:
                    self.orphan = 1

                if debug:
                    show_masks(self.content_image, self.style_image, L, R, self.K + self.orphan, show_axis=True)

                if use_dcrf:
                    L = [crf(content_image * 255, l)[0] for l in L]
                    R = [crf(style_image * 255, r)[0] for r in R]
                    if debug:
                        show_masks(self.content_image, self.style_image, L, R, self.K + self.orphan)

                self.L = L
                self.R = R
        else:
            #Get voc masks
            L_voc = load_voc_mask(voc_names[0], shape=self.content_image.shape)
            R_voc = load_voc_mask(voc_names[1], shape=self.content_image.shape)

            # Drop border and turn to binary
            self.L = toMultipleArray(pngToMaskFormat(L_voc))[0:20].astype("float32")
            self.R = toMultipleArray(pngToMaskFormat(R_voc))[0:20].astype("float32")

            self.K = self.L.shape[0]

            if debug:
                show_masks(self.content_image, self.style_image, self.L, self.R, self.K, show_axis=True, normalized=True, vmax=1)

        self.build_and_optimize()

    def build_and_optimize(self):

        # We optimize on G
        self.G = tf.Variable(self.init_image[None, ...], name="G", dtype=tf.float32)
        self.vgg_g = vgg19.Vgg19(VGG_PATH + 'vgg19.npy')
        self.vgg_g.build(self.G)

        # Get G content
        self.G_content_layer = get_layers(self.vgg_g, self.content_layer_name)

        # Get G styles
        self.G_style_layers = get_layers(self.vgg_g, self.style_layers_name)

        with tf.name_scope("train"):
            cost_content = self.content_loss()
            self.final_content_loss = self.content_loss_weight * cost_content

            if self.K > 0 or self.voc_names:
                cost_style = self.style_loss_masks()
            else:
                cost_style = self.style_loss_normal()
                #cost_style = self.style_loss_masks_dummy()

            #cost_style += self.affinity_loss()

            self.final_style_loss = self.style_loss_weight * (1 / len(self.style_layers_name)) * cost_style

            #Matting
            M = tf.to_float(getLaplacian(self.content_image))
            self.loss_affine = self.affine_loss(self.G, M, self.matting_loss)[0][0]

            self.loss_tv = self.total_variation_loss(self.G, self.tv_loss)
            #loss_tv = 0.0001

            loss = self.final_content_loss + self.final_style_loss + self.loss_affine + self.loss_tv

            global iter_count
            iter_count = 0

            train_step = tf.contrib.opt.ScipyOptimizerInterface(
                loss,
                method='L-BFGS-B',
                options={'maxiter': self.num_iter,
                         })

            self.sess.run(tf.global_variables_initializer())

            print("Start Training")
            training = train_step.minimize(self.sess, fetches=[self.final_content_loss, self.final_style_loss, self.loss_affine, self.loss_tv], loss_callback=self.callback)


        with tf.name_scope("image_out"):
            image_out = tf.clip_by_value(tf.squeeze(self.G, [0]), 0, 1)

        '''
        for i in range(0, self.num_iter):
            cost_p, _, cost_style_p, cost_content_p = self.sess.run([loss, training, cost_style, cost_content])

            if i % 300 == 0:
                print("iter = {}, total_cost = {}, cost_content = {}, cost style = {}".format(str(i),
                                                                                              cost_p,
                                                                                              cost_content_p,
                                                                                              cost_style_p))
        '''

        # Output
        image_final = self.sess.run(image_out)

        '''SMOOTH LOCAL AFFINE => CUDA ERROR'''
        '''
        content_input = np.array(self.content_image, dtype=np.float32)
        # RGB to BGR
        #content_input = content_input[:, :, ::-1]
        # H * W * C to C * H * W
        content_input = content_input.transpose((2, 0, 1))
        input_ = np.ascontiguousarray(content_input, dtype=np.float32)

        _, H, W = np.shape(input_)

        output_ = np.ascontiguousarray(image_final.transpose((2, 0, 1)), dtype=np.float32)
        best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, 15, 1e-1).transpose(1, 2, 0)
        result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
        '''

        plt.figure(figsize=(14, 12))
        plt.subplot(4, 4, 1)
        plt.imshow(self.init_image)

        plt.subplot(4, 4, 2)
        plt.imshow(self.content_image)

        plt.subplot(4, 4, 3)
        plt.imshow(self.style_image)

        plt.subplot(4, 4, 4)
        plt.imshow(image_final)

        #writer = tf.summary.FileWriter("output", self.sess.graph)
        self.image_final = image_final;

    # Return the final image, usefull for two step optimization
    def get_final(self):
        return self.image_final

    # Function at each step
    def callback(self, content_loss, style_loss, affine_loss, tv_loss):
        global iter_count;

        if iter_count % 100 == 0:
            if self.debug:
                print('Iteration {} / {} '.format(iter_count, self.num_iter))
                print('Content loss: {} / No weight: {}'.format(content_loss, content_loss / self.content_loss_weight))
                print('Style loss: {} / No weight: {}'.format(style_loss, style_loss / self.style_loss_weight))
                print('Affine loss: {} / No weight: {}'.format(affine_loss, affine_loss / self.matting_loss))
                print('Tv loss: {} / No weight: {}'.format(tv_loss, affine_loss / self.tv_loss))


        iter_count += 1

    # Input is of shape [B, H, W, C]
    def compute_gram(self, array):
        shape = array.shape
        matrix = np.reshape(array, [shape[1] * shape[2], shape[3]])
        return np.matmul(matrix.transpose(), matrix)

    # Input is of shape [B, H, W, C]
    def compute_gram_tensor(self, tensor):
        shape = tensor.get_shape()
        matrix = tf.reshape(tensor, shape=[-1, int(shape[3])])
        return tf.matmul(matrix, matrix, transpose_a=True)


    # Computes the style loss over all K for each layer
    def style_loss_masks(self):
        loss = 0
        for key in self.style_layers:
            for k in range(0, self.K + self.orphan):
                if k in self.k_style:
                    tensor = self.G_style_layers.get(key)

                    shape_g = tensor.get_shape().as_list()
                    g_mask = skimage.transform.resize(self.L[k][..., None], (shape_g[1], shape_g[2]),
                                                                         mode='constant', order=0)

                    weighted_g_layer = tf.multiply(tensor, g_mask)
                    gram_g = self.compute_gram_tensor(weighted_g_layer)
                    g_mask_mean = tf.to_float(tf.reduce_mean(g_mask))

                    # Deep photo style transfer way, I prefer my way
                    '''
                    gram_g = tf.cond(tf.greater(g_mask_mean, 0.),
                                            lambda: gram_g / (tf.to_float(tf.size(tensor)) * g_mask_mean),
                                            lambda: gram_g
                                        )
                    '''
                    layer_s = self.style_layers.get(key)
                    shape_s = layer_s.shape
                    s_mask = skimage.transform.resize(self.R[k][..., None], (shape_s[1], shape_s[2]),
                                                                          mode='constant', order=0)


                    weighted_s_layer = tf.multiply(layer_s, s_mask)
                    gram_s = self.compute_gram_tensor(weighted_s_layer)
                    s_mask_mean = tf.to_float(tf.reduce_mean(s_mask))
                    '''
                    gram_s = tf.cond(tf.greater(s_mask_mean, 0.),
                                            lambda: gram_s / (tf.to_float(layer_s.size) * s_mask_mean),
                                            lambda: gram_s
                                        )
                    '''

                    M = shape_g[3]
                    N = shape_g[1] * shape_g[2]

                    loss += tf.reduce_mean(tf.squared_difference(gram_g, gram_s)) * (1. / (4 * (N ** 2) * (M ** 2)) * ((g_mask_mean+s_mask_mean) / 2))

        return loss

    def style_loss_masks_old(self):
        loss = 0
        for key in self.style_layers:
            for k in range(0, self.K + self.orphan):
                tensor = self.G_style_layers.get(key)

                shape_g = tensor.get_shape().as_list()
                weighted_g_layer = tensor * skimage.transform.resize(self.L[k][..., None], (shape_g[1], shape_g[2]),
                                                                     mode='constant', order=0)
                gram_g = self.compute_gram_tensor(weighted_g_layer)

                layer_s = self.style_layers.get(key)

                shape_s = layer_s.shape
                weighted_s_layer = layer_s * skimage.transform.resize(self.R[k][..., None], (shape_s[1], shape_s[2]),
                                                                      mode='constant', order=0)
                gram_s = self.compute_gram(weighted_s_layer)

                M = shape_g[3]
                N = shape_g[1] * shape_g[2]

                loss += tf.reduce_mean(tf.pow((gram_g - gram_s), 2) * (1. / (4 * N ** 2 * M ** 2)))

        return loss

    # loss used with test square mask
    def style_loss_masks_dummy(self):
        loss = 0

        L = self.L[0]
        R = self.R[0]
        L = np.reshape(L, [L.shape[0], L.shape[1], 1])
        R = np.reshape(R, [R.shape[0], R.shape[1], 1])

        L = np.zeros(L.shape).astype("float32")
        L[L.shape[0] // 4:-L.shape[0] // 4, L.shape[1] // 4:-L.shape[1] // 4] = 1

        R = np.zeros(R.shape).astype("float32")
        R[R.shape[0] // 4:-R.shape[0] // 4, R.shape[1] // 4:-R.shape[1] // 4] = 1

        # L = L[:, :, 0]
        # R = R[:, :, 0]

        if (self.debug):
            plt.figure()
            plt.imshow(L[:, :, 0])
            plt.figure()
            plt.imshow(R[:, :, 0])

        for i in range(0, self.K):
            for key in self.style_layers:

                tensor = self.G_style_layers.get(key)
                shape_g = tensor.get_shape().as_list()

                # upscale masks to be same size as tensor
                currL = skimage.transform.resize(L, (shape_g[1], shape_g[2]), mode='constant', order=0)
                weighted_g_layer = tensor * currL


                gram_g = self.compute_gram_tensor(weighted_g_layer)

                layer_s = self.style_layers.get(key)
                shape_s = layer_s.shape
                currR = skimage.transform.resize(R, (shape_s[1], shape_s[2]), mode='constant', order=0)
                weighted_s_layer = layer_s * currR


                gram_s = self.compute_gram(weighted_s_layer)

                M = shape_g[3]
                N = shape_g[1] * shape_g[2]

                loss += (1/self.K) * tf.reduce_mean(tf.pow((gram_g - gram_s), 2)) * (1. / (4 * N ** 2 * M ** 2))


        return loss

    # Basic style loss, used if K = 0
    def style_loss_normal(self):
        loss = 0
        for key in self.style_layers:
            tensor = self.G_style_layers.get(key)
            shape = tensor.get_shape().as_list()

            gram_g = self.compute_gram_tensor(tensor)
            gram_s = self.grams_style.get(key)

            M = shape[3]
            N = shape[1] * shape[2]

            loss += tf.reduce_mean(tf.pow((gram_g - gram_s), 2)) * (1. / (4 * (N ** 2) * (M ** 2)))

        return loss

    # Content loss term over a layer
    def content_loss(self):
        shape = self.content_layer.get(self.content_layer_name[0]).shape
        return tf.reduce_mean(tf.pow(
            self.content_layer.get(self.content_layer_name[0]) - self.G_content_layer.get(self.content_layer_name[0]),
            2)) * 1. / (2 * shape[3] * shape[1] * shape[2])

    # When tested affinity loss MSE
    def affinity_loss(self):

        return tf.reduce_mean(tf.pow(
            self.A - self.compute_affinity_matrix_tf(self.G_content_layer.get(self.content_layer_name[0]),
                                                     tf.convert_to_tensor(
                                                         self.style_layer.get(self.content_layer_name[0]))),
            2)) / 2

    # Maps shape should be [1, H, W, C]
    def compute_affinity_matrix_tf(self, content_maps, reference_maps):

        print(content_maps)
        print(reference_maps)

        content_maps = tf.squeeze(content_maps, axis=0)
        reference_maps = tf.squeeze(reference_maps, axis=0)

        content_maps = tf.reshape(content_maps, [int(content_maps.get_shape()[0]) * int(content_maps.get_shape()[1]),
                                                 int(content_maps.get_shape()[2])])

        reference_maps = tf.reshape(reference_maps,
                                    [int(reference_maps.get_shape()[0]) * int(reference_maps.get_shape()[1]),
                                     int(reference_maps.get_shape()[2])])

        return tf.matmul(content_maps, reference_maps, transpose_b=True)

    # Photorealistic loss from deep_photo style transfer
    def affine_loss(self, output, M, weight):
        loss_affine = 0.0
        output_t = tf.to_float(output / 255.)
        for Vc in tf.unstack(output_t, axis=-1):
            Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
            loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0),
                                     tf.sparse_tensor_dense_matmul(M, tf.expand_dims(Vc_ravel, -1)))

        return loss_affine * weight

    # tv loss also from deep photo style transfer
    def total_variation_loss(self, output, weight):
        shape = output.get_shape()
        tv_loss = tf.reduce_sum((output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + \
                  (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
        return tv_loss * weight