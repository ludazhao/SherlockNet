import numpy as np
import os
import pandas as pd

'''

Test unpickling.

'''

# TODO: move the following to constants.py
OUT_PATH = '../data/tagged_images_pickled'
X_OUTFILE = 'X.npy'
METADATA_OUTFILE = 'metadata.pickle'

X = np.load(os.path.join(OUT_PATH, X_OUTFILE))
metadata = pd.read_pickle(os.path.join(OUT_PATH, METADATA_OUTFILE))

print X.shape
