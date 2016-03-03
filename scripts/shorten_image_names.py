import glob, os

os.chdir("/Users/luda/reorg3/")

for folder in glob.glob("*/"):
    os.chdir("/Users/luda/reorg3/")
    os.chdir(folder)
    for fn in glob.glob("*.jpg"):
        newname = '_'.join(fn.split('_')[:4]) + '_.jpg'
        print fn, newname
        os.system("mv '{}' '{}'".format(fn, newname))



