from skimage import io
# import numpy as np
from nd2reader import ND2Reader
import time
import os
import sys

def illumination_correction(img, avg):
    """
    Correct the illumination inhomogeneity in microscope images.
    
    Args:
    img -- input image with illumination inhomogeneity
    avg -- average of (a large number of) raw images
    
    Returns:
    corrected -- corrected image
    """
    corrected = (img / avg * img.mean() / (img / avg).mean()).astype('uint8')
    return corrected

nd2Dir = sys.argv[1]
remove = True
if len(sys.argv) > 2:
    remove = bool(int(sys.argv[2]))


folder, file = os.path.split(nd2Dir)

name, ext = os.path.splitext(file)
saveDir = os.path.join(folder, name)
saveDir8 = os.path.join(folder, name, '8-bit')
if os.path.exists(saveDir8) == False:
    os.makedirs(saveDir8)    
with open(os.path.join(saveDir, 'log.txt'), 'w') as f:
    f.write('nd2Dir = ' + str(nd2Dir) + '\n')
    
# Compute average
# to minimize the memory needed, I first loop over an nd2 file to get the average
# Then loop one more time to compute the output

threshold = 100
if remove == True:
    with ND2Reader(nd2Dir) as images:
        count = 0
        for num, image in enumerate(images):
            if image.mean() > threshold: # exclude light-off images, typically in kinetics experiment
                count += 1
                if count == 1:
                    avg = image.astype('float64')
                else:
                    avg += image
        avg = avg / count / 8          
    
with ND2Reader(nd2Dir) as images:
    for num, image in enumerate(images):
        img8 = (image/2**3).astype('uint8')
        if image.mean() > threshold and remove == True:
            corrected = illumination_correction(img8, avg)
        else:
            corrected = img8
        io.imsave(os.path.join(saveDir8, '%04d.tif' % num), corrected)
        with open(os.path.join(saveDir, 'log.txt'), 'a') as f:
            f.write(time.asctime() + ' // Frame {0:04d} converted\n'.format(num))
        
""" DESCRIPTION
Convert *.nd2 file to image sequence of 8-bit grayscale images. Save this image sequence in a subfolder under the same folder as the *.nd2 file with corresponding name as the *.nd2 file name.

This script does not apply auto-contrast and save both 16-bit and 8-bit images.

- Edit:
06162020 - No longer export 16-bit images.
08162020 - 1. write input arguments in log.txt
           2. (important) add illumination correction. All frames will be corrected according to the whole video.
08182020 - Add argument 'remove', determining if background subtraction is applied or not. Default to True.
""" 

""" SYNTAX
python to_tif.py nd2Dir remove
"""

""" TEST PARAMS
nd2Dir = E:\Github\Python\generic_proc\test_images\test.nd2
"""

""" LOG
Tue Jan 14 20:54:03 2020 // Frame 00000 converted
"""



    
