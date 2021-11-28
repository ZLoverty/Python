import os
from myImageLib import dirrec
import sys

"""
This script is used to convert all the image sequence in a given folder to a videos (avi).
The video will be saved in the parent folder of the image sequence folder.

Edit:
11242021 -- initial commit.
11262021 -- Major change: now convert all the 8-bit folder in given folder to videos
            Change name to to_vid.py
"""

"""SYNTAX
python to_vid.py folder [fmt=] [fps=]

folder -- the folder that contains the 8-bit tif image sequence,
          this script convert all 8-bit image sequences in given folder by default
fmt -- specify the format of image file name, default to %05d.tif
fps -- specify the frame rate of the output video, default to 50
"""

"""TEST PARAMS
folder = ..\test_images\to_vid
fmt = "8b%05d.tif"
"""

def main(folder, **kwargs):
    fps = 50
    fmt = "%05d.tif"
    for kw in kwargs:
        if kw == "fmt":
            fmt = kwargs[kw]
        if kw == "fps":
            fps = float(kwargs[kw])
    l = dirrec(folder, fmt % 0)
    for i in l:
        if "8-bit" in i:
            image_folder = os.path.split(i)[0]
            parent_folder, name = os.path.split(os.path.split(image_folder)[0]) # assume name/8-bit/*.tif
            output_file = os.path.join(parent_folder, "{}.avi".format(name))
            input_imseq = os.path.join(image_folder, fmt)
            cmd = 'ffmpeg -y -framerate {0:f} -i {1} -vcodec h264 {2}'.format(fps, input_imseq, output_file)
            print("==============Start converting {} to video~~==============\n".format(name))
            os.system(cmd)
            print("\nConversion of {0} is successful!\nA video is saved at {1}\n\n".format(name, output_file))

if __name__=="__main__":
    folder = sys.argv[1]
    main(folder, **dict(arg.split('=') for arg in sys.argv[2:]))
