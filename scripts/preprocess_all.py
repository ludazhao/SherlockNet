from PIL import Image
import numpy as np
import os
import glob
import csv
import multiprocessing
from multiprocessing import Pool
csv.register_dialect("textdialect", delimiter='\t')
'''

Converts images to their greyscale + cropped versions.

'''


def process_image(file, fldr):
	try: im = Image.open(file, 'r')
	except: 
		writer.writerow(["Could not open:", file])
		return
	
	try: im = im.convert('L')  # makes it greyscale
	except: 
		writer.writerow(["Could not convert to grayscale:", file])
		return
		
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
	#print data.shape
	result_width, result_height = data.shape
	filename = os.path.basename(file)
	filepath = os.path.join(fldr, '%s_postproc.jpg' % filename)
	if result_width != CROP_WIDTH or result_height != CROP_HEIGHT:
		#print 'Fail to save due to sizing error: %s' % filepath
		writer.writerow(["Sizing error", filepath])
	else:
		try:
			out_im = Image.fromarray(data, mode='L')
			out_im.save(filepath)
		except:
			writer.writerow(["Could not save", filepath])
			return
		#print 'Saved %s' % filepath


# TODO: move the following to constants.py
# DATA_DIR = '../data/toy_data_raw'
# DATA_PATH = '../data/lol'
# OUT_PATH = '../data/lol_processed'
DATA_PATH = 'G:\\MechanicalCuratorReleaseData\\extractedimagedata\\'
OUT_PATH = 'D:\\ArtHistoryNet\\images_postproc_256\\'
CROP_WIDTH, CROP_HEIGHT = 256, 256 # might want to change to 256 in future for more granularity

if __name__ == "__main__":

	os.chdir(DATA_PATH)
	try: os.mkdir(OUT_PATH)
	except: pass
	
	multiprocessing.freeze_support()
	p = Pool(4)
	
	counter = 0
	logFile = open("log_256_2.txt", 'a')
	writer = csv.writer(logFile, 'textdialect')
	for imgsize in ['plates']:
	#for imgsize in ['medium']:
		try: os.mkdir(OUT_PATH + '\\' + imgsize)
		except: pass
		
		for date in glob.glob(imgsize + '\\1*'):
			dt = int(date[-4:])
			if dt < 1893: continue  # that is because the previous run stopped there
			
			print date
			try: os.mkdir(OUT_PATH + '\\' + date)
			except: pass
			
			fldr = OUT_PATH + '\\' + date
			for file in glob.glob(date + '\\*.jpg'):
				if file.startswith('.') or not file.endswith('.jpg'): continue
				filename = os.path.basename(file)
				filepath = os.path.join(fldr, '%s_postproc.jpg' % filename)
				if glob.glob(filepath): continue  # already done
				
				p.apply_async(process_image, args=(file, fldr))
				counter += 1
				if counter % 100 == 0: print counter
	p.close()
	p.join()
			