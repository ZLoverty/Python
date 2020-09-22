import numpy as np
import os
from skimage import io
import corrLib
import sys
import time

def down_size_imseq(folder, windowsize=[50, 50], step=25):
    """
    Downsizing an image sequence of k images in given folder and save them as an numpy array of size k*m*n, 
    where m and n are the downsized dimension of each image. 
    
    Args:
    folder -- folder of image sequence
    windowsize -- parameter of corrLib.divide_windows(), pixel
    step -- parameter of corrLib.divide_windows(), pixel
    
    Returns:
    stack -- numpy array of size k*m*n
    """
    
    l = corrLib.readseq(folder)
    I_list = []
    for num, i in l.iterrows():
        img = io.imread(i.Dir)
        X, Y, I = corrLib.divide_windows(img, windowsize=windowsize, step=step)
        I_list.append(I)
    stack = np.stack(I_list, axis=0)
    
    return stack
    
arg_length = len(sys.argv)

img_folder = sys.argv[1]
out_folder = sys.argv[2]
wsize = 50
step = 25
if arg_length > 3: wsize = int(sys.argv[3])
if arg_length > 4: step = int(sys.argv[4])

if os.path.exists(out_folder) == False:
    os.makedirs(out_folder)
with open(os.path.join(out_folder, 'log.txt'), 'w') as f:
    f.write('img_folder: ' + img_folder + '\n')
    f.write('out_folder: ' + out_folder + '\n')
    f.write('wsize: ' + str(wsize) + '\n')
    f.write('step: ' + str(step) + '\n')
    f.write(time.asctime() + ' // Computation starts!\n') 
    
stack = down_size_imseq(img_folder, windowsize=[wsize, wsize], step=step)
np.save(os.path.join(out_folder, 'stack.npy'), stack)

with open(os.path.join(out_folder, 'log.txt'), 'a') as f:
    f.write(time.asctime() + ' // Computation finishes!')

""" EDIT
09202020 -- First edit
"""

""" DESCRIPTION
Concentration has always been derived from the bright field images. While images are typically 1280*1080, the data we use are always coarse-grained down to 50*42. Although the down sizing step takes only several milliseconds, repeating the process over and over for hundreds of thousands of times (all images) takes a lot of time. Thus, I decide to convert the image sequences to the coarse-grained version and save them as numpy binary data. In this way, the 3000 frame video will be down sized to 3000*50*42 float16 numbers, which in total is 6M instead of the 4.64G. By saving this down sized data, the data also gets more portable because of the small size, while all the useful information is retained.
"""

""" SYNTAX
python small_imseq.py img_folder out_folder wsize step

img_folder -- folder of image sequence
out_folder -- folder to save the numpy array
wsize -- window size, assuming square, pixel
step -- parameter of corrLib.divide_windows(), pixel

Note: the array will be saved as file 'stack.npy'
"""

""" TEST PARAMS
img_folder -- E:\Github\Python\Correlation\test_images\autocorr\80
out_folder -- E:\Github\Python\Correlation\test_images\autocorr\80\small_imseq
wsize -- 50
step -- 25
"""

""" LOG
img_folder: E:\Github\Python\Correlation\test_images\autocorr\80
out_folder: E:\Github\Python\Correlation\test_images\autocorr\80\small_imseq
wsize: 50
step: 25
Sun Sep 20 13:41:29 2020 // Computation starts!
Sun Sep 20 13:41:29 2020 // Computation finishes!
"""
