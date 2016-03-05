import os
import numpy as np
import random
from PIL import Image
from PIL import ImageEnhance

"""
Implements data augmentation as described in retrain.py > add_input_distortions() but using
Pillow image library instead of tensorflow.
"""

#Generate a number of different augmented images(flipped, cropped, scaled) at the same directory as the original image

def gen_aug_imgs(num_imgs, file_path, num_existing_augs=0, flip_left_right=False, random_crop=0, random_scale=0, random_brightness=0, verbose=False):
    im = Image.open(file_path, 'r')
    width, height = im.size
    if verbose:
        print 'random_crop=%s, random_scale=%s, random_brightness=%s' % (random_crop, random_scale, random_brightness)
    for i in xrange(num_existing_augs, num_existing_augs + num_imgs):
        margin_scale = 1.0 + (random_crop / 100.0)
        resize_scale = 1.0 + (random_scale / 100.0)
        margin_scale_value = margin_scale
        resize_scale_value = random.uniform(1.0, resize_scale)
        #print 'resize_scale_val=%s' % resize_scale_value
        scale_value = margin_scale_value * resize_scale_value
        #print 'scale_value=%s' % scale_value
        precrop_width = int(scale_value * width)
        precrop_height = int(scale_value * height)
        #print 'precrop_width=%s' % precrop_width
        #print 'precrop_height=%s' % precrop_height
        precrop_im = im.resize((precrop_width, precrop_height), Image.BICUBIC)
        bounding_box = get_rand_bounding_box(precrop_width, precrop_height, width, height)
        #print bounding_box
        cropped_im = precrop_im.crop(bounding_box)
        if flip_left_right:
            flipped_im = cropped_im.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            flipped_im = cropped_im
        brightness_min = 1.0 - (random_brightness / 100.0)
        brightness_max = 1.0 + (random_brightness / 100.0)
        #brightness_max = 1.0
        enhancer = ImageEnhance.Brightness(flipped_im)
        brightness_value = random.uniform(brightness_min, brightness_max)
        #print 'brightness_val=%s' % brightness_value
        brightened_im = enhancer.enhance(brightness_value)
        # brightened_im = flipped_im.point(lambda p: p * brightness_value)

        data = np.asarray(brightened_im)
        file_path_no_ext = os.path.splitext(file_path)[0]
        aug_file_path = '%s_AUG_%d.jpg' % (file_path_no_ext, i)
        out_im = Image.fromarray(data, mode='L')
        out_im.save(aug_file_path)
        if verbose:
            print 'Saved %s' % aug_file_path


def get_rand_bounding_box(width, height, crop_width, crop_height):
    top = random.randint(0, height - crop_height)
    bottom = top + crop_height
    left = random.randint(0, width - crop_width)
    right = left + crop_width
    return (left, top, right, bottom)


# def test_run():
#     DATA_PATH = '/Users/kywang/Downloads/test_data_aug'
#     for file in os.listdir(DATA_PATH):
#         if file.startswith('.') or not file.endswith('.jpg') or '_AUG_' in file:
#             continue
#         file_path = os.path.join(DATA_PATH, file)
#         num_existing_augs = 0
#         for rc in xrange(0, 100, 20):
#             for rs in xrange(0, 100, 20):
#                 for rb in xrange(0, 100, 20):
#                     gen_aug_imgs(1,file_path,num_existing_augs,False,random_crop=rc,random_scale=rs,random_brightness=rb)
#                     num_existing_augs += 1
# test_run()
