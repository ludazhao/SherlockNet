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

from run_inference import predict_star, predict, f_star

def create_graph(pb_file):
    """"Creates a graph from saved GraphDef file and returns a Graph object.
    Returns:
    Graph holding the trained Inception network.
    """
    model_filename = pb_file
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    
# from http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb-with-numpy.html
def to_rgb(im):
    w, h = im.shape[:2]
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

# from http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb-with-numpy.html
def to_rgbs(im):
    n, w, h = im.shape[:3]
    ret = np.empty((n, w, h, 3), dtype=np.uint8)
    ret[:, :, :, 0] = im[:,:,:,0]
    ret[:, :, :, 1] = im[:,:,:,0]
    ret[:, :, :, 2] = im[:,:,:,0]
    return ret

# from https://gist.github.com/yusugomori/4462221
def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2
        
        

    
if __name__ == "__main__":
        
        
        
    # ### TEST ###
    # p = mp.Pool(8)
    # print zip([1,2,3],[4,5,6])
    # print p.map(f_star, itertools.izip([1,2,3], itertools.repeat(4)))
    # exit()
    
    #graph = create_graph"/data/retrain_manualtags/output_graph_best.pb")
    create_graph("/data/output_graphfinal.pb")
    sess = tf.Session()

    params = pickle.load(open("/data/10k_aug_outputs/output_params_lr1e-3_adam9800.pkl", 'r'))
    fw = params["final_weights"]
    fb = params["final_biases"]

    labels = []
    with open("/data/10k_aug_outputs/output_labels9800.txt", 'r') as ifile:
        for line in ifile:
            labels.append(line.rstrip())

                
    # read in hdf5 file
    print "Reading in hdf5"
    image_hdf5 = h5py.File('/data/image_data.hdf5','r')
    (image_metadata, book_metadata, image_to_idx) = pickle.load(open("/data/all_metadata_w11ktags.pkl", 'r'))

    manager = mp.Manager()
    images_to_scores = manager.dict()
    args = {"images_to_scores": images_to_scores, 
            "image_to_idx": image_to_idx, 
            "image_hdf5": image_hdf5, 
            "fw": fw, 
            "fb": fb, 
            "sess": sess}
            
            
    p = mp.Pool(8)
    print "Timing one"
    a = time.time()
    
    ctr = 0
    for img in image_to_idx:
        ctr += 1
        if ctr == 2: break
        try:
            predict_star((img, args))
        except:
            continue
    b = time.time()
    print b - a
    
    
    print "Timing 8 in parallel"
    a = time.time()
    
    ctr = 0
    p.map(predict_star, itertools.izip(image_to_idx.keys()[:10], itertools.repeat(args)))
    p.close()
    p.join()
    b = time.time()
    print b - a
    #pickle.dump(images_to_scores, open("image_to_scores_10k.pkl", 'w'))

    #a = mp.Process(target=predict_star, args=((images_for_tagging[7], args)))










