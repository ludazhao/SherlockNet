import os
import numpy
from PIL import Image

#Generate a number of different augmented images(flipped, cropped, scaled) at the same directory as the original image
def gen_aug_imgs(num_imgs, file_path, flipped=True, crop_per=20, scale_per=20):

