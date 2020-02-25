from xiaolei.chain.tracking import distance_filter_frame
import sys
import os
import pandas as pd
import time
# filter out spurious trajectories in a colloidal chain, based on minimal number of neighbors criterion.


input_csv = sys.argv[1]
output_csv = sys.argv[2]
if len(sys.argv) > 3:
    crit_dist = int(sys.argv[3])
    neighbors = int(sys.argv[4])
else:
    crit_dist = 70
    neighbors = 2
    
output_folder, file = os.path.split(output_csv)
if os.path.exists(output_folder) == 0:
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
    pass
    
traj = pd.read_csv(input_csv)
new_traj = pd.DataFrame()
for frame in traj.frame.drop_duplicates():
    traj1 = traj.loc[traj.frame==frame]
    traj1_filt = distance_filter_frame(traj1, crit_dist=crit_dist, neighbors=neighbors)
    new_traj = new_traj.append(traj1_filt)
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // Frame ' + str(frame) + ' calculated\n')
new_traj.to_csv(output_csv, index=False)
    
""" TEST COMMAND
python dist_filt.py input_csv output_csv crit_dist neighbors
"""
        
"""  TEST PARAMS
input_csv = R:\Dip\DNA_chain\fluorescent\center_of_mass_chain\test1.csv
output_csv = R:\Dip\DNA_chain\fluorescent\center_of_mass_chain\test1_filt.csv
"""

""" LOG
Thu Feb 13 11:39:47 2020 // 900-901 calculated
Thu Feb 13 11:40:35 2020 // 902-903 calculated
"""

""" SPEED 51 s/frame
Tue Feb 11 14:27:52 2020 // 900-901 calculated
Tue Feb 11 14:28:41 2020 // 902-903 calculated
"""