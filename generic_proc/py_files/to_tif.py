from skimage import io
import numpy as np
from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import math
import pdb
from scipy import fftpack
from myImageLib import dirrec, to8bit, bpass
import trackpy as tp
import time
import os
import sys

nd2Dir = sys.argv[1]
folder, file = os.path.split(nd2Dir)

name, ext = os.path.splitext(file)
saveDir = os.path.join(folder, name)
saveDir16 = os.path.join(folder, name, '16-bit')
saveDir8 = os.path.join(folder, name, '8-bit')
if os.path.exists(saveDir16) == False:
    os.makedirs(saveDir16)
if os.path.exists(saveDir8) == False:
    os.makedirs(saveDir8)
    
with open(os.path.join(saveDir, 'log.txt'), 'w') as f:
    pass
    
with ND2Reader(nd2Dir) as images:
    for num, image in enumerate(images):
        img8 = (image/2**3).astype('uint8')
        io.imsave(os.path.join(saveDir16, '%04d.tif' % num), image)
        io.imsave(os.path.join(saveDir8, '%04d.tif' % num), img8)
        with open(os.path.join(saveDir, 'log.txt'), 'a') as f:
            f.write(time.asctime() + ' // Frame {0:04d} converted\n'.format(num))
        
""" DESCRIPTION
Convert *.nd2 file to image sequence of 8-bit grayscale images. Save this image sequence in a subfolder under the same folder as the *.nd2 file with corresponding name as the *.nd2 file name.

This script does not apply auto-contrast and save both 16-bit and 8-bit images.
""" 

""" SYNTAX
python to_tif.py nd2Dir
"""

""" TEST PARAMS
nd2Dir = I:\Github\Python\generic_proc\test_images\test.nd2
"""

""" LOG
Tue Jan 14 20:54:03 2020 // Frame 00000 converted
"""



    
