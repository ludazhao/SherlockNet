import os.path
import re
import sys
import tarfile
import time
import multiprocessing as mp
#from mpi4py import MPI
#import mpi4py
import itertools

#import tensorflow.python.platform
#from six.moves import urllib
import numpy as np
#import tensorflow as tf
import h5py
import glob
import cPickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from tensorflow.python.platform import gfile
import collections

#from run_inference import predict_star, predict
import pandas as pd
import seaborn as sns


# coding=UTF-8
import nltk
from nltk.corpus import brown

# This is a fast and simple noun phrase extractor (based on NLTK)
# Feel free to use it, just keep a link back to this post
# http://thetokenizer.com/2013/05/09/efficient-way-to-extract-the-main-topics-of-a-sentence/
# Create by Shlomi Babluki
# May, 2013

# This is our fast Part of Speech tagger
#############################################################################
brown_train = brown.tagged_sents(categories='news')
regexp_tagger = nltk.RegexpTagger(
    [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
     (r'(-|:|;)$', ':'),
     (r'\'*$', 'MD'),
     (r'(The|the|A|a|An|an)$', 'AT'),
     (r'.*able$', 'JJ'),
     (r'^[A-Z].*$', 'NNP'),
     (r'.*ness$', 'NN'),
     (r'.*ly$', 'RB'),
     (r'.*s$', 'NNS'),
     (r'.*ing$', 'VBG'),
     (r'.*ed$', 'VBD'),
     (r'.*', 'NN')
])
unigram_tagger = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(brown_train, backoff=unigram_tagger)
#############################################################################


# This is our semi-CFG; Extend it according to your own needs
#############################################################################
cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"
#############################################################################


class NPExtractor(object):

    def __init__(self, sentence):
        self.sentence = sentence

    # Split the sentence into singlw words/tokens
    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens

    # Normalize brown corpus' tags ("NN", "NN-PL", "NNS" > "NN")
    def normalize_tags(self, tagged):
        n_tagged = []
        for t in tagged:
            if t[1] == "NP-TL" or t[1] == "NP":
                n_tagged.append((t[0], "NNP"))
                continue
            if t[1].endswith("-TL"):
                n_tagged.append((t[0], t[1][:-3]))
                continue
            if t[1].endswith("S"):
                n_tagged.append((t[0], t[1][:-1]))
                continue
            n_tagged.append((t[0], t[1]))
        return n_tagged

    # Extract the main topics from the sentence
    def extract(self):

        tokens = self.tokenize_sentence(self.sentence)
        tags = self.normalize_tags(bigram_tagger.tag(tokens))

        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = "%s+%s" % (t1[1], t2[1])
                value = cfg.get(key, '')
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = "%s %s" % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break

        matches = []
        for t in tags:
            if t[1] == "NNP" or t[1] == "NNI":
            #if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN":
                matches.append(t[0])
                
        return matches
        
######################
######################
###### MY CODE #######
######################
######################


# 1. get list of indices for each tag
print "Getting list of indices for each tag"
(image_metadata, book_metadata, image_to_idx) = pickle.load(open("/data/all_metadata_1M_tags.pkl", 'r'))
indices = collections.defaultdict(lambda: [])  # list of indices of images that represent animals

for idx in xrange(970218):
    tag = image_metadata[idx][-1]
    #if tag == 'animals': continue  # we already ran these
    #if len(indices[tag]) < 50: indices[tag].append(idx)
    indices[tag].append(idx)
    
print indices.keys()

# 2. noun phrases
ocr_hdf5 = h5py.File('/data/ocr_data.hdf5','r')
print "Extracting noun phrases"

def get_noun_phrases_from_img(chunk, img):
    chunk = int(chunk)
    img = int(img)
    
    phrase = ' '.join(ocr_hdf5['Chunk{}'.format(chunk)][img][1:]).decode("ascii", errors="ignore")
    np_extractor = NPExtractor(phrase)
    multiword_res = [x.lower() for x in np_extractor.extract()]
    
    res = []
    for word in multiword_res:
        res.extend(word.split(' '))
        
    # number of items in the set divided by total length; a marker of English or not-English
    pct_np = float(len(set(res)))/len(phrase)
    
    # get the words that appear most often
    #print Counter(res).most_common(10)

    return pct_np, res

def worker_by_tag(tag):
    print "Extracting noun phrases for tag {}".format(tag)
    idx_to_noun_phrases = {}
    for idx in indices[tag]:
        chk = idx/5000
        i = idx % 5000
        try:
            pct_np, res = get_noun_phrases_from_img(chk, i)
            idx_to_noun_phrases[idx] = set([x for x in res if len(x)>3])
        except:
            idx_to_noun_phrases[idx] = set([])
    
    pickle.dump(idx_to_noun_phrases, open("/data/nearest_neighbor_tagging/idx_to_noun_phrases_{}.pkl".format(tag), 'w'))

# for tag in indices:
#     worker(tag)
    
def worker_by_10k(start):
    print "Extracting noun phrases from {} to {}".format(start, start + 10000)
    idx_to_noun_phrases = {}
    
    ctr = 0
    for idx in range(start, start + 10000):
        ctr += 1
        #if ctr == 50: break
        chk = idx/5000
        i = idx % 5000
        try:
            pct_np, res = get_noun_phrases_from_img(chk, i)
            idx_to_noun_phrases[idx] = set([x for x in res if len(x)>3])
        except:
            idx_to_noun_phrases[idx] = set([])
    
    pickle.dump(idx_to_noun_phrases, open("/data/nearest_neighbor_tagging/idx_to_noun_phrases_{}.pkl".format(start), 'w'))
    
p = mp.Pool(8)
#p.map(worker_by_10k, range(0, 970218, 10000))
#p.map(worker_by_10k, [90000, 70000, 110000, 240000, 340000, 460000, 160000, 230000, 530000])

# get unmapped ones
chunks_to_process = []
chunks = range(0,970218,10000)
for chk in chunks:
    fn = "/data/nearest_neighbor_tagging/idx_to_noun_phrases_{}.pkl".format(chk)
    if not glob.glob(fn):
        chunks_to_process.append(chk)
    else:
        try:
            a = pickle.load(open(fn, 'r'))
        except:
            chunks_to_process.append(chk)
    
    
print chunks_to_process
p.map(worker_by_10k, chunks_to_process)

# for i in chunks_to_process:
#     print i
#     worker_by_10k(i)
# for tag in indices:
#     worker(tag)
#     
