from pivLib import PIV1
import sys
import os
from corrLib import readseq
from skimage import io
import time

input_folder = sys.argv[1]
output_folder = sys.argv[2]
winsize = int(sys.argv[3])
overlap = int(sys.argv[4])
fps = int(sys.argv[5])

if os.path.exists(output_folder) == 0:
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
    pass
    
dt = 1 / fps

l = readseq(input_folder)
for num, i in l.iterrows():
    if num % 2 == 0:
        I0 = io.imread(i.Dir)
        continue 
    I1 = io.imread(i.Dir)
    frame_data = PIV1(I0, I1, winsize, overlap, dt)
    frame_data.to_csv(os.path.join(output_folder, i.Name+'.csv'), index=False)
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + i.Name + ' calculated\n')

""" TEST COMMAND
python piv_imseq.py input_folder output_folder winsize overlap fps
"""
        
"""  TEST PARAMS
input_folder = I:\Github\Python\PIV\test_images\imseq
output_folder = I:\Github\Python\PIV\test_images\imseq\pivData
winsize = 50
overlap = 25
fps = 30
"""

""" LOG
Mon Feb 10 17:43:35 2020 // 0001 calculated
Mon Feb 10 17:43:36 2020 // 0003 calculated
1 frame/s
""" 