import random
import sys
from image_util import *

# TODO: move to constants.py
FLIP_PROB = 0.3
NUM_AUGS_PER = 3
NUM_EXISTING_AUGS_PER = 0
RANDOM_CROP = 20
RANDOM_SCALE = 20
RANDOM_BRIGHTNESS = 20

#PARENT_DIR = '../data/reorg3_img_aug'
subdirs = ['Animals', 'Architecture', 'Decorations', 'Diagrams', 'Landscape', 'Maps', 'Miniatures',
           'Nature', 'Objects', 'People', 'Seals', 'Text']

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'Please pass in root of folder subtree containing tagged images'
        print 'Usage example: python augment_data.py ../data/reorg3_img_aug'
        sys.exit(0)
    parent_dir = sys.argv[1]

    flip_left_right = False
    if random.random() < FLIP_PROB:
        flip_left_right = True
    for subdir in subdirs:
        DATA_PATH = os.path.join(parent_dir, subdir)
        for file in os.listdir(DATA_PATH):
            if file.startswith('.') or not file.endswith('.jpg') or '_AUG_' in file:
                continue
            file_path = os.path.join(DATA_PATH, file)
            gen_aug_imgs(num_imgs=NUM_AUGS_PER,
                         file_path=file_path,
                         num_existing_augs=NUM_EXISTING_AUGS_PER,
                         flip_left_right=flip_left_right,
                         random_crop=RANDOM_CROP,
                         random_scale=RANDOM_SCALE,
                         random_brightness=RANDOM_BRIGHTNESS,
                         verbose=True)