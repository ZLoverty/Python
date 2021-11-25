import cv2
from skimage import io
import os
from corrLib import readdata
from myImageLib import to8bit
import time
import sys

"""
This script is used to convert an image sequence to a video (mp4).
The video will be saved in the parent folder of the image sequence folder.

Edit:
11242021 -- initial commit.
"""

folder = sys.argv[1]
l = readdata(folder, 'tif')
sample_img = io.imread(l.Dir[0])
input_video = os.path.join(os.path.split(folder)[0], '{}.avi'.format(os.path.split(folder)[1]))
output_video = os.path.join(os.path.split(folder)[0], '{}.mp4'.format(os.path.split(folder)[1]))
video = cv2.VideoWriter(input_video, cv2.VideoWriter_fourcc(*'DIVX'), 50, sample_img.shape, isColor=False)
for num, i in l.iterrows():
    img = io.imread(i.Dir)
    video.write(to8bit(img))
video.release()

os.system('ffmpeg -i {0} {1}'.format(input_video, output_video))

"""SYNTAX
python imseq_to_mp4.py folder

folder -- the folder that contains the tif image sequence
"""

"""TEST PARAMS
folder = ..\test_images\img_to_vid
"""
