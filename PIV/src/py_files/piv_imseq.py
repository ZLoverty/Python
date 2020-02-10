from pivLib import imseqPIV
import sys
import os

# folder = r'I:\Google Drive\Code\Python\PIV\test_images\imseq'
# winsize = 50 # pixels
# overlap = 25 # pixels
# dt = 0.033 # frame interval (sec)

input_folder = sys.argv[1]
output_folder = sys.argv[2]
winsize = int(sys.argv[3])
overlap = int(sys.argv[4])
fps = int(sys.argv[5])

if os.path.exists(output_folder) == 0:
    os.makedirs(output_folder)
    
dt = 1 / fps


    
data = imseqPIV(folder, winsize, overlap, dt)

data.to_csv(os.path.join(output_folder