import os, csv, glob, collections
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import argparse
plt.style.use('ggplot')
csv.register_dialect("textdialect", delimiter=',')
mpl.rcParams['font.family'] = 'Arial'

from shutil import copyfile

DATADIR = '../data/'

csv_files = ['tags_brian_part.csv', 'tags_luda_part.csv', 'tags_karen_part.csv']

url_to_tag = {}
id_to_tag = {}
id_to_keyword = {}

for f in csv_files:
    f = DATADIR + f
    with open(f, 'r') as ifile:
        reader = csv.reader(ifile ,'textdialect')
        headers = reader.next()
        for row in reader:
            if '1' in row:
                tag = headers[row.index('1')]
                url_to_tag[row[0]] = tag
                id_to_tag[row[0].split('/')[5]] = tag
                id_to_keyword[row[0].split('/')[5]] = row[13]

print "Number of images:", len(url_to_tag)

count = 0
csv.register_dialect("textdialect", delimiter='\t')

id_to_image_file = {}
images_dat = []
#Now to extract from harddrive
direc = "/Volumes/My Passport/MechanicalCuratorReleaseData/imagedirectory/imagedirectory-master"
for fn in glob.glob(direc + "/*.tsv"):
    #if 'plates' in fn or 'unknown' in fn: continue
    with open(fn, 'r') as ifile:
        reader = csv.reader(ifile ,'textdialect')
        header = reader.next()
        #0 is volume, 4 is BL_DLS_ID(not used), 6 is book identifier, 8 is date, 10 is image_idx, 11 is page, 12 is flickr_id
        for row in reader:
            if row[12] in id_to_tag:
                file_prefix = row[6].zfill(9) + '_' + row[0] + '_' + row[11].zfill(6) + '_' + row[10]
                if 'small' in fn:
                    dir1 = 'embellishments'
                elif 'medium' in fn:
                    dir1 = 'medium'
                elif 'plates' in fn:
                    dir1 = 'plates'
                images_dat.append([dir1, str(row[8]), file_prefix, row[12]])
print "Found " + str(len(images_dat)) + " images."
DIRIMAGE = "/Volumes/My Passport/MechanicalCuratorReleaseData/extractedimagedata/"
DIRCOPY = "/Users/luda/Dropbox/CS231N/ArtHistoryNet/data/tagged/"
#get the images and copy to data directory

f1 = open('image_to_tags.csv', 'wb')
f2 = open('id_to_image.csv', 'wb')
f3 = open('id_to_keywords.cvs', 'wb')
f1writer = csv.writer(f1, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
f2writer = csv.writer(f2, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
f3writer = csv.writer(f3, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
f3writer.writerows(id_to_keyword)
for im in images_dat:
    impath = DIRIMAGE + im[0] + '/' + im[1] + '/'
    for fn in os.listdir(impath):
        if fn.startswith(im[2]):
            print "copied image"
            copyfile(impath + fn, DIRCOPY + fn)
            f1writer.writerow([fn, id_to_tag[im[3]]])
            f2writer.writerow([im[3], fn])










