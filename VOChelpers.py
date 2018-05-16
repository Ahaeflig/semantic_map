import pandas as pd
import os
import glob
from bs4 import BeautifulSoup
from more_itertools import unique_everseen
import numpy as np
import PIL
from PIL import Image
import sklearn
import skimage
from matplotlib import pyplot as plt

'''
Helpers file with function useful to work with VOC Pascal
'''

root_dir = "data/VOC/VOCdevkit/VOC2012/"
img_dir = os.path.join(root_dir, 'JPEGImages/')
seg_class_dir = os.path.join(root_dir, 'SegmentationClass/')
ann_dir = os.path.join(root_dir, 'Annotations')
set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
segmentation_dir = os.path.join(root_dir, 'ImageSets', 'Segmentation')

N_CLASSES = 22


def list_image_segmentation(dataset="train"):
    filename = os.path.join(segmentation_dir, dataset + ".txt")
    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        header=None)
    return df


# Loads image and its segmentation
# Return image and mask as array (pair) in range [0,1]
def load_pair_voc(img_name, shape):

    img_file = os.path.join(img_dir, img_name + ".jpg")
    img = PIL.Image.open(img_file)
    img.thumbnail(shape, Image.ANTIALIAS)
    img = img.resize((shape[1], shape[0]), Image.ANTIALIAS)

    mask_file = os.path.join(seg_class_dir, img_name + ".png")
    mask = PIL.Image.open(mask_file)
    mask = mask.resize((shape[1], shape[0]), PIL.Image.NEAREST)

    return [img, mask]

# Load mask from voc image
def load_voc_mask(img_name, shape):
    mask_file = os.path.join(seg_class_dir, img_name + ".png")
    image_mask = PIL.Image.open(mask_file)
    mask = image_mask.resize((shape[1], shape[0]), PIL.Image.NEAREST)
    return mask

# Load mask from voc image
def load_voc_img(img_name, shape):
    img_file = os.path.join(img_dir, img_name + ".jpg")
    img = PIL.Image.open(img_file)
    img.thumbnail(shape, Image.ANTIALIAS)
    img = img.resize((shape[1], shape[0]), Image.ANTIALIAS)
    return img

# Turns a png image into the proper segmentation mask with values [0, N_CLASSES+1]
def pngToMaskFormat(img):
    return np.array(img.convert(mode="P", palette=Image.ADAPTIVE, colors=N_CLASSES))


# Returns a (height, width, nclasses) array with 1 on pixel where it belongs to one of the 21 classes
# argument comes from pngToMaskFormat
def toMultipleArray(PconvertedImage):
    final_ = np.zeros((N_CLASSES, PconvertedImage.shape[0], PconvertedImage.shape[1]))
    for i in range(PconvertedImage.shape[0]):
        for j in range(PconvertedImage.shape[1]):
            pixel_val = PconvertedImage[i, j]
            if pixel_val < N_CLASSES:
                final_[pixel_val, i, j] = 1
            else:
                # Border
                final_[N_CLASSES - 1, i, j] = 1

    return final_

def computeAcc(test, model):
    #First binarize
    mask = np.where(test > 0.5, 1.0, 0.0)
    return sklearn.metrics.accuracy_score(model.flatten(), mask.flatten())


'''
Stuff from git
Stuff from git
Stuff from git
'''

def list_image_sets():
    """
    List all the image sets from Pascal VOC. Don't bother computing
    this on the fly, just remember it. It's faster.
    """
    return [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']


def annotation_file_from_img(img_name):
    """
    Given an image name, get the annotation file for that image
    Args:
        img_name (string): string of the image name, relative to
            the image directory.
    Returns:
        string: file path to the annotation file
    """
    return os.path.join(ann_dir, img_name) + '.xml'


def load_annotation(img_filename):
    """
    Load annotation file for a given image.
    Args:
        img_name (string): string of the image name, relative to
            the image directory.
    Returns:
        BeautifulSoup structure: the annotation labels loaded as a
            BeautifulSoup data structure
    """
    xml = ""
    with open(annotation_file_from_img(img_filename)) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml, "html5lib")


def load_data_multilabel(data_type=None):
    """
    Returns a data frame for all images in a given set in multilabel format.

    Args:
        data_type (string, optional): "train" or "val"

    Returns:
        pandas DataFrame: filenames in multilabel format
    """
    if data_type is None:
        raise ValueError('Must provide data_type = train or val')
    filename = os.path.join(set_dir, data_type + ".txt")
    cat_list = list_image_sets()
    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        header=None,
        names=['filename'])
    # add all the blank rows for the multilabel case
    for cat_name in cat_list:
        df[cat_name] = 0
    for info in df.itertuples():
        index = info[0]
        fname = info[1]
        anno = load_annotation(fname)
        objs = anno.findAll('object')
        for obj in objs:
            obj_names = obj.findChildren('name')
            for name_tag in obj_names:
                tag_name = str(name_tag.contents[0])
                if tag_name in cat_list:
                    df.at[index, tag_name] = 1
    return df

