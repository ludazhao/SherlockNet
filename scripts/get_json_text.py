import json
import cPickle as pickle
import sys
import multiprocessing as mp
import os

def worker(start, end):
	print start, end
	jsondir = "/data/json/"
	
	for i in range(20):	
		name_to_text = {}
		newstart = start + (i * 10000)
		newend = newstart + 10000
		print start, newstart, newend
		if newstart > len(name_to_book_info.keys()): break

		ctr = 0
		for index in name_to_book_info.keys()[newstart:newend]:
			if ctr % 1000 == 0: 
				print "Size:", newstart, ctr, sys.getsizeof(name_to_text)
			info = name_to_book_info[index]
			name = info[0]
			fn = jsondir + "{}/{}_{}_text.json".format(info[2], info[1], info[3])
			try:
				with open(fn) as ifile:
					data = json.load(ifile)
					name_to_text[index] = [name, '', '', '']
					name_to_text[index][1] = data[info[4]][1]
					name_to_text[index][2] = data[info[4]+1][1]
					name_to_text[index][3] = data[info[4]+2][1]
			except:
				pass
			ctr += 1
		
		pickle.dump(name_to_text, open("/data/json/name_to_text/name_to_text_{}.pkl".format(newstart), 'w'))
	
	

if __name__ == '__main__':
	name_to_book_info = pickle.load(open("/data/index_to_book_info.pkl", 'r'))
	# book, book_cat, vol, page

	#os.mkdir("/data/json/name_to_text/")
	#name_to_text = {}
	#failure = "/Users/bdo/Documents/SherlockNet/cannot_find_json.txt"

	procs = []
	for i in range(5):
		p = mp.Process(target=worker, args=(i*200000,(i+1)*200000,))
		p.start()
		procs.append(p)
	for p in procs:	
		p.join()	
