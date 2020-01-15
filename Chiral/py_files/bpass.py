import numpy as np
from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import math
import skimage.io
import pdb
from scipy import fftpack
from myImageLib import dirrec
import trackpy as tp
import os
import sys

bpassLow = 1
bpassHigh = 200
nd2Dir = sys.argv[1]
if len(sys.argv) > 2:        
    bpassLow = sys.argv[2]
    bpassHigh = sys.argv[3]
folder, file = os.path.split(nd2Dir)
name, ext = os.path.splitext(file)
saveDir = os.path.join(folder, name)
if os.path.exists(saveDir) == False:
    os.mkdir(saveDir)
with ND2Reader(nd2Dir) as images:
    print('Processing ' + nd2Dir)
    for num, image in enumerate(images):
        img8 = to8bit(image)
        img_bpass = bpass(img8, bpassLow, bpassHigh)
        skimage.io.imsave(os.path.join(saveDir, '%5d.tif' % num), img_bpass)


    
