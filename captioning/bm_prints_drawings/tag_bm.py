# a python version of the ipynb notebook running a modfieid InceptionNet for tagging using the BM dataset

import numpy as np
import tensorflow as tf
import os
import json


ROOT_DIR = '/data/images_validation_10k_combined/'
MODEL_ROOT = '/tmp/'
#ROOT_DIR = '/data/captioning/img_100_gs/'
#ROOT_DIR = '/Users/luda/Projects/SherlockNet/img_100_gs/'
#MODEL_ROOT = '/Users/luda/Projects/SherlockNet/tmp/boat/'


def create_graph(modelFullPath):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_images(imageFns, c, tag):
    print "category: ", c
    print "tag: ", tag
    res = []
#     if not tf.gfile.Exists(imagePath):
#         tf.logging.fatal('File does not exist %s', imagePath)
#         return answer
    modelFullPath = MODEL_ROOT + tag + '/output_graph.pb'
    labelsFullPath = MODEL_ROOT + tag + '/output_labels.txt'

    f = open(labelsFullPath, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]
    # Creates graph from saved GraphDef.
    create_graph(modelFullPath)
    res.append((len(imageFns), labels[0], labels[1]))

    with tf.Session() as sess:
        count = 0
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for imageFn in imageFns:
            count +=1
            print imageFn
            image_data = tf.gfile.FastGFile(ROOT_DIR + c + '/'  + imageFn, 'rb').read()
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-2:][::-1]  # Getting 2 scores
            res.append((imageFn, float(predictions[0]), float(predictions[1])))
            if (count % 100 == 0):
                print count

    return res, labels

tags = ['boat', 'mountain', 'river', 'bridge']
categories = ['nature', 'landscapes']
#tags = ['boat']
for c in categories:
    for tag in tags:
        tf.reset_default_graph()
        res, labels = run_inference_on_images(os.listdir(ROOT_DIR + c), c, tag)
        with open('/data/captioning/bm_tagging_' + c + '_' + tag + '.json', 'w') as outfile:
            json.dump(res, outfile)