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
import sys

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
		
	chk_start = int(sys.argv[1])
	
	logfile = open("/data/1M_tags/{}_log.txt".format(chk_start), 'w')
	
	
	# ### TEST ###
	# p = mp.Pool(8)
	# print zip([1,2,3],[4,5,6])
	# print p.map(f_star, itertools.izip([1,2,3], itertools.repeat(4)))
	# exit()
	
	#graph = create_graph"/data/retrain_manualtags/output_graph_best.pb")
	create_graph("/data/classify_image_graph_def.pb")
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

	softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
	for chk in range(chk_start, chk_start + 25):
		chunk = "Chunk" + str(chk)
		if chunk not in image_hdf5: continue
		print chunk
		logfile.write("{}\n".format(chunk))
		
		idx_to_arr = {}
		for i in range(10000):
			print "\t{}".format(i)
			logfile.write("\t{}".format(i))

			try:
				a = [to_rgb(image_hdf5["Chunk0"][i][:,:,0])]
				with tf.device("/cpu:0"):
					predictions = sess.run(softmax_tensor, {'ExpandDims:0': a})[:,0,0,:]
				preds = np.dot(predictions, fw) + fb
				preds = np.array([softmax(preds[i]) for i in range(preds.shape[0])])
				idx_to_arr[i] = preds
				f = time.time()
			except:
				continue
				
		pickle.dump(idx_to_arr, "/data/1M_tags/{}.pkl".format(chunk))