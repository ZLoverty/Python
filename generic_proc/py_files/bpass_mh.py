import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import io
import pdb
from scipy import fftpack
from myImageLib import dirrec, to8bit, bpass
import trackpy as tp
import time
import os
import sys
import corrLib as cl

bpassLow = 3
bpassHigh = 500
folder = sys.argv[1]
saveDir = sys.argv[2]
if len(sys.argv) > 3:        
    bpassLow = int(sys.argv[3])
    bpassHigh = int(sys.argv[4])

if os.path.exists(saveDir) == False:
    os.mkdir(saveDir)

with open(os.path.join(saveDir, 'log.txt'), 'w') as f:
    pass
    
l = cl.readseq(folder)
for num, i in l.iterrows():
    img8 = io.imread(i.Dir)
    img_bpass = bpass(img8, bpassLow, bpassHigh)
    img_bp_mh = cl.match_hist(img_bpass, img8)
    io.imsave(os.path.join(saveDir, '%04d.tif' % num), img_bp_mh)
    with open(os.path.join(saveDir, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // Frame {0:04d} converted\n'.format(num))
        
""" DESCRIPTION
Convert *.nd2 file to image sequence and apply bandpass filter to each image. 
Save this image sequence in a subfolder under the same folder as the *.nd2 file with corresponding name as the *.nd2 file name.
An additional histogram matching is performed so that the processed image looks similar to the original image. 
""" 

""" SYNTAX
python bpass_mh.py folder saveDir bpassLow bpassHigh
"""

""" TEST PARAMS
folder = E:\Github\Python\Correlation\test_images\ixdiv_autocorr\raw 
saveDir = E:\Github\Python\Correlation\test_images\ixdiv_autocorr\bp_mh
bpassLow = 3
bpassHigh = 500
"""

""" LOG
Tue Apr 28 17:20:25 2020 // Frame 0000 converted
Tue Apr 28 17:20:27 2020 // Frame 0001 converted
Tue Apr 28 17:20:28 2020 // Frame 0002 converted
Tue Apr 28 17:20:30 2020 // Frame 0003 converted
Tue Apr 28 17:20:32 2020 // Frame 0004 converted
Tue Apr 28 17:20:33 2020 // Frame 0005 converted
"""