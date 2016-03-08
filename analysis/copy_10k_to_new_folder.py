import os
import glob
import cPickle as pickle
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

print "Loading data"
(image_metadata, book_metadata, image_to_idx) = pickle.load(open("D:/ArtHistoryNet/data/all_metadata.pkl", 'r'))
image_to_tag = pickle.load(open("D:/ArtHistoryNet/scripts/image_to_tags_10k.pkl", 'r'))
if not glob.glob("D:/ArtHistoryNet/images_validation_10k"): os.mkdir("D:/ArtHistoryNet/images_validation_10k")

basedir = "D:/ArtHistoryNet/images_postproc_256/"

print "Moving files"
counter = 0
for img in image_to_tag:
    tag = image_to_tag[img].capitalize()
    idx = image_to_idx[img]
    md = image_metadata[idx]
    date = md[1]
    
    if md[2] == 'e': size = "embellishments"
    elif md[2] == 'm': size = "medium"
    else: size = "plates"

    print basedir + size + '/' + date + '/' + img + "*", tag
    fn = glob.glob(basedir + size + '/' + date + '/' + img + "*")
    print fn
    continue
    if counter > 10: break
    else:
        fn = fn[0]
        newfolder = "D:/ArtHistoryNet/images_validation_10k/" + tag
        if not glob.glob(newfolder): os.mkdir(newfolder)
        newfn = newfolder + "/" + img + ".jpg"
        os.system("MOVE \"{}\" \"{}\"".format(fn, newfn))
		