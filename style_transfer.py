import tensorflow as tf
import numpy as np
import skimage
import skimage.transform

from helpers import *

from helpers import getLaplacian

# IO
import matplotlib.pyplot as plt

# Datastructure
from collections import OrderedDict

# Custom
import vgg19
import vgg19_atrous

VGG_PATH = 'data/vgg/'
DATA_PATH = 'data/'


class StyleTransfer:
    def __init__(self, content_layer_name, style_layers_name, mask_layer_name, init_image, content_image, style_image, session, num_iter,
                 content_style_loss_ratio, K=15, normalize=True, debug=False, orphan=True, use_dcrf=True, matting_loss=1000, tv_loss=1000):

        # Right now we only want to use same sizes
        assert (content_image.shape == style_image.shape)

        self.K = K

        self.content_layer_name = content_layer_name
        self.style_layers_name = style_layers_name
        self.mask_layer_name = mask_layer_name

        self.init_image = init_image
        self.content_image = content_image
        self.style_image = style_image
        self.num_iter = num_iter

        self.sess = session
        self.lambda_ = content_style_loss_ratio

        self.matting_loss = matting_loss
        self.tv_loss = tv_loss

        self.debug = debug

        ''' 
        =======================================
        Prepare constants || Initialize
        =======================================
        '''

        # Vgg
        # Only one graph is needed cause we assume the shape of all the images are the same
        self.vgg = vgg19.Vgg19(VGG_PATH + 'vgg19.npy')
        self.vgg_shape = [1, content_image.shape[0], content_image.shape[1], content_image.shape[2]]

        images = tf.placeholder("float", self.vgg_shape, name="images")
        self.vgg.build(images)

        # Content from content image and style image remember eval_layers return dicts
        self.content_layer = eval_layers(self.sess, self.content_image, self.vgg, self.content_layer_name)
        self.style_layer = eval_layers(self.sess, self.style_image, self.vgg, self.content_layer_name)

        # Mask layers,  note that eval_layers expects array of names
        self.mask_content_layer = eval_layers(self.sess, content_image, self.vgg, [self.mask_layer_name])
        self.mask_style_layer = eval_layers(self.sess, style_image, self.vgg, [self.mask_layer_name])

        '''
        self.vgg_atrous = vgg19_atrous.Vgg19_Atrous(VGG_PATH + 'vgg19.npy')
        self.vgg_atrous.build(images)
        
        self.mask_content_layer = eval_layers(self.sess, content_image, self.vgg_atrous, [self.mask_layer_name])
        self.mask_style_layer = eval_layers(self.sess, style_image, self.vgg_atrous, [self.mask_layer_name])
        '''

        # Style from style image, precomputed since the gram doesn't change
        self.style_layers = eval_layers(self.sess, style_image, self.vgg, self.style_layers_name)
        self.grams_style = {layer_name: self.compute_gram(layer) for layer_name, layer in self.style_layers.items()}

        # Masks between the content layer of content image and style image
        self.A = compute_affinity_matrix(self.mask_content_layer.get(self.mask_layer_name),
                                         self.mask_style_layer.get(self.mask_layer_name))


        if self.K > 0:
            L, R = get_masks(K, self.A, self.mask_content_layer.get(self.mask_layer_name).shape[1:3],
                             self.mask_style_layer.get(self.mask_layer_name).shape[1:3], orphan=orphan, normalize=normalize)

            self.orphan = 0

            #Add orphan masks
            if orphan:
                self.orphan = 1


            if debug:
                show_masks(self.content_image, self.style_image, L, R, self.K + self.orphan, show_axis=True)

            if use_dcrf:
                L = [dcrf_refine(content_image * 255, l) for l in L]
                R = [dcrf_refine(style_image * 255, r) for r in R]

                if debug:
                    show_masks(self.content_image, self.style_image, L, R, self.K + self.orphan)

            self.L = L
            self.R = R

        self.build_and_optimize()

    def build_and_optimize(self):

        # We optimize on G
        self.G = tf.Variable(self.init_image[None, ...], name="G", dtype=tf.float32)

        #init_image = np.random.randn(1, self.content_image.shape[0], self.content_image.shape[1], 3).astype(np.float32) * 0.0001
        #self.G = tf.Variable(init_image, name="G", dtype=tf.float32)

        self.vgg_g = vgg19.Vgg19(VGG_PATH + 'vgg19.npy')

        self.vgg_g.build(self.G)

        # Get G content
        self.G_content_layer = get_layers(self.vgg_g, self.content_layer_name)

        # Get G styles
        self.G_style_layers = get_layers(self.vgg_g, self.style_layers_name)

        with tf.name_scope("train"):
            cost_content = self.content_loss()
            self.final_content_loss = self.lambda_ * cost_content

            if self.K > 0:
                cost_style = self.style_loss_masks()
            else:
                cost_style = self.style_loss_normal()
                #cost_style = self.style_loss_masks_dummy()

            #cost_style += self.affinity_loss()
            #cost_style = self.style_loss_masks_dummy()

            self.final_style_loss =  (1 - self.lambda_) * (1 / len(self.style_layers_name)) * cost_style

            #Matting
            M = tf.to_float(getLaplacian(self.content_image))
            self.loss_affine = self.affine_loss(self.G, M, self.matting_loss)[0][0]

            self.loss_tv = self.total_variation_loss(self.G, self.tv_loss)
            #loss_tv = 0.0001

            loss = self.final_content_loss + self.final_style_loss + self.loss_affine + self.loss_tv

            # optimizer = tf.train.AdamOptimizer()
            # gvs = optimizer.compute_gradients(loss)
            # training = optimizer.apply_gradients(gvs)

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

        plt.figure(figsize=(12, 10))
        plt.subplot(3, 3, 1)
        plt.imshow(self.content_image)

        plt.subplot(3, 3, 2)
        plt.imshow(self.style_image)

        plt.subplot(3, 3, 3)
        plt.imshow(image_final)

        #writer = tf.summary.FileWriter("output", self.sess.graph)


    def callback(self, content_loss, style_loss, affine_loss, tv_loss) :
        global iter_count;

        if iter_count % 100 == 0:
            if self.debug:
                print('Iteration {} / {}'.format(iter_count, self.num_iter))
                print('Content loss: {}'.format(content_loss))
                print('Style loss: {}'.format(style_loss))
                print('Affine loss: {}'.format(affine_loss))
                print('Tv loss: {}'.format(tv_loss))

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

    def style_loss_masks(self):
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

                loss += (1 / self.K) * tf.reduce_sum(tf.pow((gram_g - gram_s), 2)) * (1. / (4 * N ** 2 * M ** 2))

        return loss

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

                loss += (1/self.K) * tf.reduce_sum(tf.pow((gram_g - gram_s), 2)) * (1. / (4 * N ** 2 * M ** 2))


        return loss

    def style_loss_normal(self):
        loss = 0
        for key in self.style_layers:
            tensor = self.G_style_layers.get(key)
            # shape = tensor.get_shape().as_list()
            shape = tensor.get_shape().as_list()

            gram_g = self.compute_gram_tensor(tensor)
            gram_s = self.grams_style.get(key)

            # gram_s_t = tf.convert_to_tensor(gram_s)

            M = shape[3]
            N = shape[1] * shape[2]

            # loss += tf.nn.l2_loss(gram_g - gram_s) * 1./(4 * N**2 * M**2)
            # loss += tf.reduce_sum(tf.pow((gram_g - gram_s), 2)) * (1. / (4 * N ** 2 * M ** 2))
            loss += tf.reduce_sum(tf.pow((gram_g - gram_s), 2) * (1. / (4 * (N ** 2) * (M ** 2))) )

        return loss

    def content_loss(self):

        shape = self.content_layer.get(self.content_layer_name[0]).shape
        print(shape);

        return tf.reduce_sum(tf.pow(
            self.content_layer.get(self.content_layer_name[0]) - self.G_content_layer.get(self.content_layer_name[0]),
            2) * 1. / (2 * shape[3] * shape[1] * shape[2]) )

    def affinity_loss(self):

        return tf.reduce_sum(tf.pow(
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

    def affine_loss(self, output, M, weight):
        loss_affine = 0.0
        output_t = tf.to_float(output / 255.)
        for Vc in tf.unstack(output_t, axis=-1):
            Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
            loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0),
                                     tf.sparse_tensor_dense_matmul(M, tf.expand_dims(Vc_ravel, -1)))

        return loss_affine * weight

    def total_variation_loss(self, output, weight):
        shape = output.get_shape()
        tv_loss = tf.reduce_sum((output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + \
                  (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
        return tv_loss * weight