import os
import glob
import numpy as np
import cPickle as pickle
import h5py
from scipy.misc import imsave

print "Loading data"
(image_metadata, book_metadata, image_to_idx) = pickle.load(open("/data/all_metadata.pkl", 'r'))
if not glob.glob("/data/decorations_by_date"): os.mkdir("/data/decorations_by_date")
image_hdf5 = h5py.File('/data/image_data.hdf5','r')
labels = []
with open("/data/10k_aug_outputs/output_labels9800.txt", 'r') as ifile:
    for line in ifile:
        labels.append(line.rstrip())
print labels

basedir = "/data/decorations_by_date/"

print "Moving files"
counter = 0
for i in range(100):
	chunk_file = "/data/1M_tags/Chunk{}.pkl".format(i)
	print chunk_file
	scores = pickle.load(open(chunk_file, 'r'))
	
	for idx in range(len(scores.keys())):
		tag = labels[np.argmax(scores[idx])]
		image_metadata[i * 5000 + idx][-1] = tag
		if tag == 'decorations':
			[img, date] = image_metadata[i * 5000 + idx][:2]
			date = int(date) 
			
			newfolder = basedir + str(10 * (date/10))  # HOW GRANULAR??
			if not glob.glob(newfolder): os.mkdir(newfolder)
			newfn = newfolder + "/" + img + ".jpg"

			imsave(newfn, image_hdf5["Chunk{}".format(i)][idx][:,:,0])  
			
pickle.dump((image_metadata, book_metadata, image_to_idx), open("/data/all_metadata_1M_tags.pkl", 'w'))