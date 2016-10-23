import os.path
import re
import sys
import tarfile
import time
import multiprocessing as mp
import itertools
from collections import Counter

import numpy as np
import h5py
import glob
import cPickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import collections

import pandas as pd
import seaborn as sns

print "Loading noun phrases"
idx_to_noun_phrases = pickle.load(open("/data/idx_to_noun_phrases.pkl", 'r'))

import enchant
import difflib
eng = enchant.Dict("en_US")
br = enchant.Dict("en_GB")
fr = enchant.Dict("fr_FR")
sp = enchant.Dict("es")
ger = enchant.Dict("de_DE")

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def process_word(word):
    if eng.check(word):  # spelled correctly
        if word in stop_words: return ""
        return word
    else:  # take the top ranked suggestion
        try:
            new_word = br.suggest(word)[0].lower()
            new_score = enchant.utils.levenshtein(word, new_word)
            if new_score > 3: return ""  # too far away
        except: return ""  # if there are no suggestions from the English dictionary, the word is probably malformed
        if new_word in stop_words: return "" 
            
        # other languages
        if fr.check(word) or ger.check(word) or sp.check(word): return ""
        
        return new_word

p = mp.Pool(63)
nearest_image_files = glob.glob("/data/nearest_neighbor_tagging/nearest_neighbors/*.pkl")
for fn in nearest_image_files:
    if 'top20' in fn: continue  
    if glob.glob(fn[:-4] + "_top20.pkl"): continue
    
    image_to_top_words = {}
    ctr = 0
    
    idx_to_neighbors = pickle.load(open(fn, 'r'))
    for idx in idx_to_neighbors:
        ctr += 1
        if ctr % 1000 == 0: print str(datetime.now()), fn, ctr
        
        # get words from all neighbors
        all_words = list(idx_to_noun_phrases[idx])
        for neighbor in idx_to_neighbors[idx]:
            all_words.extend(idx_to_noun_phrases[neighbor[0]])  # neighbor = (idx, score)
        all_words_set = list(set(all_words))
        
        # check for misspellings, stopwords, etc and stem the word
        results = p.map(process_word, all_words_set)
        
        results_stemmed = [stemmer.stem(str(x)) for x in results]
        unstem = collections.defaultdict(lambda: [])
        for x in zip(results_stemmed, results):
            if x[0] != x[1]: unstem[x[0]].append(str(x[1]))  # collect words that match the stem            
            
        corrected = dict(zip(all_words_set, results_stemmed))
        all_words_pared = [corrected[word] for word in all_words if corrected[word] != ""]
        
        final_tags = list(Counter(all_words_pared).most_common(20))
        if len(final_tags) > 0:
            for i in range(len(final_tags)):
                word_tuple = list(final_tags[i])
                if word_tuple[0] in unstem:   # if multiple words match to a stem we want those
                    word_tuple.append(list(set(unstem[word_tuple[0]])))
                    final_tags[i] = tuple(word_tuple)
            
        image_to_top_words[idx] = final_tags
    pickle.dump(image_to_top_words, open(fn[:-4] + "_top20.pkl", 'w'))
