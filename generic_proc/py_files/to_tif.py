from skimage import io
# import numpy as np
from nd2reader import ND2Reader
import time
import os
import sys
import shutil

def to8bit(img16):
    """
    Enhance contrast and convert to 8-bit
    """
    # if img16.dtype != 'uint16':
        # raise ValueError('16-bit grayscale image is expected')
    maxx = img16.max()
    minn = img16.min()
    img8 = (img16 - minn) / (maxx - minn) * 255
    return img8.astype('uint8')

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

def disk_capacity_check(file):
    """Check if the capacity of disk is larger than twice of the file size.
    Args:
    file -- directory of the (.nd2) file being converted
    Returns:
    flag -- bool, True if capacity is enough.
    """
    fs = os.path.getsize(file) / 2**30
    ds = shutil.disk_usage(file)[2] / 2**30
    print("File size {0:.1f} GB, Disk size {1:.1f} GB".format(fs, ds))
    return ds > 2 * fs

nd2Dir = sys.argv[1]
remove = False
if len(sys.argv) > 2:
    remove = bool(int(sys.argv[2]))

# disk capacity check
if disk_capacity_check(nd2Dir) == False:
    print("No enough disk capacity!")
    exit()

print("DISK CAPACITY OK")
folder, file = os.path.split(nd2Dir)

name, ext = os.path.splitext(file)
saveDir = os.path.join(folder, name, 'raw')
saveDir8 = os.path.join(folder, name, '8-bit')
if os.path.exists(saveDir) == False:
    os.makedirs(saveDir)
with open(os.path.join(saveDir, 'log.txt'), 'w') as f:
    f.write('nd2Dir = ' + str(nd2Dir) + '\n')
if os.path.exists(saveDir8) == False:
    os.makedirs(saveDir8)
with open(os.path.join(saveDir8, 'log.txt'), 'w') as f:
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
        # img8 = (image/2**3).astype('uint8')
        # 8-bit image now are only used for visualization, i.e. convert to videos
        io.imsave(os.path.join(saveDir8, '{:05d}.tif'.format(num)), to8bit(image))
        if image.mean() > threshold and remove == True:
            corrected = illumination_correction(image, avg)
        else:
            corrected = image
        io.imsave(os.path.join(saveDir, '%05d.tif' % num), corrected, check_contrast=False)
        with open(os.path.join(saveDir, 'log.txt'), 'a') as f:
            f.write(time.asctime() + ' // Frame {0:04d} converted\n'.format(num))

""" DESCRIPTION
Convert *.nd2 file to image sequence of 8-bit grayscale images. Save this image sequence in a subfolder under the same folder as the *.nd2 file with corresponding name as the *.nd2 file name.

This script does not apply auto-contrast and save both 16-bit and 8-bit images.

- Edit:
06162020 - No longer export 16-bit images.
08162020 - 1. write input arguments in log.txt
           2. (important) add illumination correction. All frames will be corrected according to the whole video.
08182020 - Add argument 'remove', determining if background subtraction is applied or not. Default to False.
10282021 - Remove "8-bit" folder, export original images instead of converting to 8-bit
11042021 - 1. Set `check_contrast` to False to avoid CLI spamming
           2. Add 'exp1' before the image number, in accordance to Cristian's image naming convention
11262021 - 1. Remove the 'exp1' flag at the beginning of each image file
           2. Add saturated 8-bit image output for visualization (this means we need a big overhead of disk space!)
11302021 - Add disk_capacity_check function to avoid running out disk space
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
