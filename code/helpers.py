# Utils
import skimage
import skimage.io
import skimage.transform
import numpy as np

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