from VOChelpers import *
from helpers import *
import tensorflow as tf
import vgg19
import vgg19_atrous
import math
import sklearn

class MIL:
    def __init__(self):

        self.VGG_PATH = 'data/vgg/'
        '''
        Load VOC data ~20 seconds
        '''
        #self.df_train = load_data_multilabel("train")
        self.df_val = load_data_multilabel("val")

        '''
        Setup VGG
        '''
        tf.reset_default_graph()
        self.sess = tf.Session()

        vgg_shape = [1, None, None, 3]
        images = tf.placeholder("float", vgg_shape, name="images")
        '''
        self.vgg = vgg19.Vgg19(self.VGG_PATH + 'vgg19.npy')
        self.vgg.build(images)
        '''
        self.vgg = vgg19_atrous.Vgg19_Atrous(VGG_PATH + 'vgg19.npy')
        self.vgg.build(images)

        #self.setImage(self, pathToImageToLabel)




    def setImage(self, pathToImageToLabel):
        '''
        Setup image to label
        '''
        #print(pathToImageToLabel + " set")
        self.imageToLabel = load_image2(pathToImageToLabel)
        self.imageToLabel5_5 = eval_layers(self.sess, self.imageToLabel, self.vgg, ["conv5_4"]).get("conv5_4")
        self.imageToLabelTargetShape = self.imageToLabel5_5.shape


    def imgToAffinityReshaped(self, image):
        dict_content = eval_layers(self.sess, image, self.vgg, ["conv5_4"])

        #Resize
        #currentImageLayer = self.sess.run(tf.image.resize_images(dict_content.get("conv5_4"),
        #                        [self.imageToLabelTargetShape[0], self.imageToLabelTargetShape[1]]))
        return compute_affinity_matrix(self.imageToLabel5_5, dict_content.get("conv5_4"))


    #image names is an array of voc image_names
    def performLabelling(self, image_names):

        assert(self.imageToLabel is not None)

        images = [load_voc_img(name, self.imageToLabel.shape) for name in image_names]

        affinities = np.array([self.imgToAffinityReshaped(np.array(im) / 255) for im in images])
        # Stack affinities
        final_aff = np.concatenate(affinities, axis=1)

        # Drop border and load masks
        maskLoad = [load_voc_mask(name, (self.imageToLabel5_5.shape[1], self.imageToLabel5_5.shape[2])) for name in image_names]

        masks_converted = np.array([toMultipleArray(pngToMaskFormat(mask)) for mask in maskLoad])

        # Stack masks
        mask = np.concatenate(masks_converted, axis=1)

        mask = mask.reshape((N_CLASSES, -1))
        mask = mask[0:21]

        W = np.array([softmax(final_aff[i, None]) for i in range(final_aff.shape[0])])

        prediction = np.array([W[i].dot(np.transpose(mask)) for i in range(W.shape[0])])

        predictionReshape = prediction[:, 0, :].reshape((self.imageToLabelTargetShape[1], self.imageToLabelTargetShape[2], N_CLASSES-1))

        #H, W, C
        crfIm = self.imageToLabel * 255
        L = [crf(crfIm, l) for l in predictionReshape.transpose((2, 0, 1))]

        predictionReshape = skimage.transform.resize(predictionReshape, (self.imageToLabel.shape[0], self.imageToLabel.shape[1]),
                                                                     mode='constant', order=0)

        return L, predictionReshape

    # Accuracy
    def computeAcc(self, test, model):
        # First binarize
        mask = np.where(test > 0.5, 1.0, 0.0)

        mask = mask.flatten()
        model = model.flatten()

        tp = float((model * mask).sum())
        fp = (mask * (1 - model)).sum()
        fn = (model * (1 - mask)).sum()

        return tp / (tp + fp + fn)
        #return sklearn.metrics.roc_auc_score(model.flatten(), mask.flatten())


    # Returns all filename from a certain category
    def retCatFileNames(self, category, dataset="val"):
        valData = self.df_val

        # Keep only element of the category
        valData = valData[valData[category] == 1]

        df_segm = pd.read_csv("data/VOC/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", header=None,
                              names=["filename"])

        df_f = df_segm.merge(pd.DataFrame(valData['filename']), how="inner")
        filenames = df_f['filename'][None, :][0]

        return filenames


    # Compute val acc for a category
    def computeCategoryAcc(self, category, numberOfImg, dataset="val", use_crf=True):
        valData = self.df_val

        # Keep only element of the category
        valData = valData[valData[category] == 1]

        df_segm = pd.read_csv("data/VOC/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", header=None, names=["filename"])

        df_f = df_segm.merge(pd.DataFrame(valData['filename']), how="inner")
        filenames = df_f['filename'][None, :][0]

        acc_sum = 0
        for i in range(len(filenames)):
            filename = filenames[i]
            filenames2 = np.copy(filenames)
            filenames2 = np.delete(filenames2, i)
            imageNamesForLabelling = np.random.choice(filenames2, numberOfImg)

            self.setImage("data/VOC/VOCdevkit/VOC2012/JPEGImages/" + filename + ".jpg")
            pred, pred_no_crf = self.performLabelling(imageNamesForLabelling)

            if use_crf:
                maskCorrect = load_voc_mask(filename, (self.imageToLabel.shape[0], self.imageToLabel.shape[1]))
            else:
                maskCorrect = load_voc_mask(filename, (self.imageToLabelTargetShape[1], self.imageToLabelTargetShape[2]))

            maskCToP = toMultipleArray(pngToMaskFormat(maskCorrect))

            vocMask = np.swapaxes(np.swapaxes(maskCToP, 0, 1), 1, 2)
            cat_index = list_image_sets().index(category)

            if use_crf:
                pred_final =pred[cat_index+1][0]
            else:
                pred_final =  pred_no_crf[:,:,cat_index+1]

            mask_final = maskCToP[cat_index+1]

            acc_sum += self.computeAcc(pred_final, mask_final)

        acc_sum /= len(filenames)
        return acc_sum

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

