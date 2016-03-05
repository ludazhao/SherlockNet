import os.path
import re
import sys
import tarfile
import time
import multiprocessing as mp
import itertools

import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
import h5py
import glob
import cPickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python.platform import gfile

# from http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb-with-numpy.html
def to_rgb(im):
    w, h = im.shape[:2]
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

# from https://gist.github.com/yusugomori/4462221
def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2
		
def run_inference_on_image(image, fileType, fw, fb, sess, visualize=True, verbose=True):
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.

    softmax_tensor = sess.graph.get_tensor_by_name('pool_3/_reshape:0')
    
    if fileType=='jpg':
        if not gfile.Exists(image):
            tf.logging.fatal('File does not exist %s', image)
        image_data = gfile.FastGFile(image).read()
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    elif fileType=='arr':
        image_data = [to_rgb(image)]
        predictions = sess.run(softmax_tensor, {'ExpandDims:0': image_data})
    else:
        print "Must be jpg or arr."
        return

    preds = softmax((np.dot(predictions, fw) + fb)[0])
    top_k = preds.argsort()[-12:][::-1]
    
    for node_id in top_k:
      score = preds[node_id]
      if verbose:
            print '%s (score = %.5f)' % (labels[int(node_id)], score)
       
    if visualize:
        if fileType=='jpg': plt.imshow(mpimg.imread(image), cmap=mpl.cm.gray)
        else:
            plt.imshow(image, cmap=mpl.cm.gray)

    return preds

	
def predict(img, params):
    num = params["image_to_idx"][img]
    chunk_num = "Chunk" + str(num / 5000)
    row_num = num % 5000
    preds = run_inference_on_image(image=params["image_hdf5"][chunk_num][row_num][:,:,0], 
                                   fileType="arr", 
                                   fw=params["fw"],
                                   fb=params["fb"],
                                   sess=params["sess"],
                                   visualize=False, verbose=False)
    params["images_to_scores"][img] = preds

def predict_star(tup):
    return predict(*tup)