import numpy as np
from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import math
import skimage.io
from scipy import fftpack
from myImageLib import dirrec, to8bit, bpass
import trackpy as tp
import time
import os
import sys

bpassLow = 2
bpassHigh = 100
nd2Dir = sys.argv[1]
if len(sys.argv) > 2:        
    bpassLow = int(sys.argv[2])
    bpassHigh = int(sys.argv[3])
folder, file = os.path.split(nd2Dir)

name, ext = os.path.splitext(file)
saveDir = os.path.join(folder, name)
if os.path.exists(saveDir) == False:
    os.mkdir(saveDir)

with open(os.path.join(saveDir, 'log.txt'), 'w') as f:
    pass
with ND2Reader(nd2Dir) as images:
    for num, image in enumerate(images):
        img8 = to8bit(image)
        img_bpass = bpass(img8, bpassLow, bpassHigh)
        skimage.io.imsave(os.path.join(saveDir, '%04d.tif' % num), img_bpass)
        with open(os.path.join(saveDir, 'log.txt'), 'a') as f:
            f.write(time.asctime() + ' // Frame {0:04d} converted\n'.format(num))
        
""" DESCRIPTION
Convert *.nd2 file to image sequence and apply bandpass filter to each image. Save this image sequence in a subfolder under the same folder as the *.nd2 file with corresponding name as the *.nd2 file name.
""" 

""" SYNTAX
python bpass.py nd2Dir bpassLow bpassHigh
"""

""" TEST PARAMS
nd2Dir = E:\Github\Python\generic_proc\test_images\test.nd2
bpassLow = 3
bpassHigh = 100
"""

""" LOG
Tue Jan 14 20:54:03 2020 // Frame 00000 converted
"""



    
