from PIL import Image
import numpy as np
import os
import pandas as pd
import pickle

'''

Converts and saves image dataset as numpy arrays.

'''

# TODO: move the following to constants.py
DATA_PATH = '../data/tagged_images_postproc'
OUT_PATH = '../data/tagged_images_pickled'
WIDTH = 128
HEIGHT = 128
X_OUTFILE = 'X.npy'
METADATA_OUTFILE = 'metadata.pickle'

files = []
for file in os.listdir(DATA_PATH):
    if file.startswith('.') or not file.endswith('.jpg'):
        continue
    files.append(file)

print 'There are %s datapoints.' % len(files)

X = np.empty((len(files), WIDTH, HEIGHT))
filename_to_metadata = {}
for idx, file in enumerate(files):
    im = Image.open(os.path.join(DATA_PATH, file), 'r')
    data = np.asarray(im)
    # w, h = data.shape
    X[idx,:,:] = data
    filename = os.path.splitext(file)[0]
    filename_to_metadata[filename] = {}

s = pd.Series(filename_to_metadata)
#print s
print X.shape
np.save(os.path.join(OUT_PATH, X_OUTFILE), X)
#pickle.dump(s, os.path.join(OUT_PATH, METADATA_OUTFILE))
s.to_pickle(os.path.join(OUT_PATH, METADATA_OUTFILE))

