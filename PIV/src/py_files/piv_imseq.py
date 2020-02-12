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
    f.write('Params\n')
    f.write('winsize: ' + str(winsize) + '\n')
    f.write('overlap: ' + str(overlap) + '\n')
    f.write('fps: ' + str(fps) + '\n')
    
dt = 1 / fps

l = readseq(input_folder)

k = 0 # serve as a flag for I0 and I1

for num, i in l.iterrows():
    # if num % 2 == 0:
        # I0 = io.imread(i.Dir)
        # continue 
    if k % 2 == 0:
        I0 = io.imread(i.Dir)
        n0 = i.Name
        k += 1
    else:
        I1 = io.imread(i.Dir)
        k += 1
        frame_data = PIV1(I0, I1, winsize, overlap, (int(i.Name)-int(n0))*dt)
        frame_data.to_csv(os.path.join(output_folder, n0 + '-' + i.Name+'.csv'), index=False)
        with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
            f.write(time.asctime() + ' // ' + n0 + '-' + i.Name + ' calculated\n')

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
Tue Feb 11 10:47:33 2020 // 0000-0001 calculated
Tue Feb 11 10:47:35 2020 // 0002-0003 calculated
.5 frame/s
""" 