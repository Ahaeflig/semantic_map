from VOChelpers import *
from helpers import *
import tensorflow as tf
import vgg19
import math


class MIL:
    def __init__(self, pathToImageToLabel, divisor=6):

        self.VGG_PATH = 'data/vgg/'
        '''
        Load VOC data ~20 seconds
        '''
        # self.df_train = load_data_multilabel("train")
        # self.df_val = load_data_multilabel("val")


        '''
        Setup VGG
        '''
        tf.reset_default_graph()
        self.sess = tf.Session()

        self.vgg = vgg19.Vgg19(self.VGG_PATH + 'vgg19.npy')
        vgg_shape = [1, None, None, 3]

        images = tf.placeholder("float", vgg_shape, name="images")
        self.vgg.build(images)

        '''
        Setup image to label
        '''
        self.imageToLabel = load_image2(pathToImageToLabel, width=800)
        self.divisor = divisor
        self.imageToLabelTargetShape = (math.floor(self.imageToLabel.shape[0]/self.divisor), math.floor(self.imageToLabel.shape[1]/self.divisor), self.imageToLabel.shape[2])
        self.imageToLabel5_5 = eval_layers(self.sess, self.imageToLabel, self.vgg, ["conv5_4"]).get("conv5_4")
        self.imageLabelReshaped = self.sess.run(tf.image.resize_images(self.imageToLabel5_5,
                                        [self.imageToLabelTargetShape[0], self.imageToLabelTargetShape[1]]))



    def imgToAffinityReshaped(self, image):
        dict_content = eval_layers(self.sess, image, self.vgg, ["conv5_4"])

        #Resize
        currentImageLayer = self.sess.run(tf.image.resize_images(dict_content.get("conv5_4"),
                                [self.imageToLabelTargetShape[0], self.imageToLabelTargetShape[1]]))

        return compute_affinity_matrix(self.imageLabelReshaped, currentImageLayer)


    #image names is an array of voc image_names
    def performLabelling(self, image_names):
        # Get masks

        imagesFinal = [load_pair_voc(name, self.imageToLabelTargetShape) for name in image_names]
        images = [load_voc_img(name, self.imageToLabel.shape) for name in image_names]

        #Todo can use zip
        affinities = np.array([self.imgToAffinityReshaped(np.array(im) / 255) for im in images])

        # Stack affinities
        final_aff = np.concatenate(affinities, axis=1)

        masks = [mask for im, mask in imagesFinal]
        masks_converted = [toMultipleArray(pngToMaskFormat(mask)) for mask in masks]

        #stacked_masks = self.stackPILMask(masks)

        #get masks to 21 dimension
        #mask = toMultipleArray(pngToMaskFormat(stacked_masks))

        #Stack masks
        mask = np.concatenate(masks_converted, axis=1)

        plt.imshow(mask[0])

        mask = mask.reshape((N_CLASSES, -1))
        print("mask.shape", mask.shape)
        print("final_aff.shape", final_aff.shape)


        #W = np.array([softmax(final_aff[None, i, :]) for i in range(final_aff.shape[0])])
        W = np.array([softmax(final_aff[i, None]) for i in range(final_aff.shape[0])])
        print("W.shape", W.shape)

        prediction = np.array([W[i].dot(np.transpose(mask)) for i in range(W.shape[0])])
        print("prediction.shape", prediction.shape)

        print(mask.shape)
        predictionReshape = prediction[:, 0, :].reshape((self.imageToLabelTargetShape[0], self.imageToLabelTargetShape[1], N_CLASSES))

        return predictionReshape



    def computeCategoryAcc(self, category, numberOfImg, divisor):
        valData = load_data_multilabel("val")

        # Keep only element of the category
        valData = valData[valData[category] == 1]


        return valData

    '''
       def stackPILMask(self, masks):
           widths, heights = zip(*(i.size for i in masks))


           width = sum(widths)
           total_height = heights[0]

           new_mask = Image.new('RGB', (width, total_height))

           x_offset = 0
           for m in masks:
               new_mask.paste(m, (x_offset, 0))
               x_offset += m.size[0]



           width = widths[0]
           total_height = sum(heights)

           new_mask = Image.new('P', (width, total_height))

           print(np.array(new_mask).dtype)
           print(np.array(masks[0]).dtype)

           y_offset = 0
           for m in masks:
               new_mask.paste(m, (0, y_offset))
               y_offset += m.size[1]

           print(np.array(new_mask).shape)
           #plt.imshow(new_mask)
           plt.imshow(masks[0])
           print(np.unique(new_mask))

           return new_mask
       '''

