from PIL import Image
import numpy as np
import os

# DATA_DIR = '../data/toy_data_raw'
# DATA_PATH = '../data/lol'
# OUT_PATH = '../data/lol_processed'
DATA_PATH = '/Users/kywang/Desktop/tagged_images'
OUT_PATH = '/Users/kywang/Desktop/tagged_images_postproc'
CROP_WIDTH, CROP_HEIGHT = 128, 128 # might want to change to 256 in future for more granularity

for file in os.listdir(DATA_PATH):
    if file.startswith('.') or not file.endswith('.jpg'):
        continue
    im = Image.open(os.path.join(DATA_PATH, file), 'r')
    im = im.convert('L')  # makes it greyscale

    width, height = im.size # resize image and crop
    if width <= height:
        new_height = int(height * (float(CROP_WIDTH)/width))
        im = im.resize((CROP_WIDTH, new_height), Image.BICUBIC)
        left, right = 0, CROP_WIDTH
        top, bottom = (new_height - CROP_HEIGHT)/2, (new_height + CROP_HEIGHT)/2
        while bottom - top < CROP_HEIGHT:
            bottom += 1
    elif height < width:
        new_width = int(width * (float(CROP_HEIGHT)/height))
        im = im.resize((new_width, CROP_HEIGHT), Image.BICUBIC)
        top, bottom = 0, CROP_HEIGHT
        left, right = (new_width - CROP_WIDTH)/2, (new_width + CROP_WIDTH)/2
        while right - left < CROP_WIDTH:
            right += 1
    im = im.crop((left, top, right, bottom))

    data = np.asarray(im)
    #print data
    print data.shape
    result_width, result_height = data.shape
    filename = os.path.splitext(file)[0]
    filepath = os.path.join(OUT_PATH, '%s_postproc.jpg' % filename)
    if result_width != CROP_WIDTH or result_height != CROP_HEIGHT:
        print 'Fail to save due to sizing error: %s' % filepath
    else:
        out_im = Image.fromarray(data, mode='L')
        out_im.save(filepath)
        print 'Saved %s' % filepath