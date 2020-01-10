from skimage import io, util
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import os
from myImageLib import dirrec
import pdb
import trackpy as tp

def threshold(img, percentile=1):
    threshold = np.percentile(img, percentile)
    imgt = img.copy()
    imgt[img<=threshold] = 255
    imgt[img>threshold] = 0
    return imgt
def locate_particle(img, **kwargs):
    area = (250, 10000)
    ar = (0.2, 1)
    thres_percentile = 1
    for kw in kwargs:
        if kw == 'area':
            area = kwargs[kw]
        elif kw == 'ar':
            ar = kwargs[kw]
        elif kw == 'thres_percentile':
            thres_percentile = kwargs[kw]
    imgt = threshold(img, percentile=thres_percentile)
    data = []
    
    label_image = skimage.measure.label(imgt)
    for region in skimage.measure.regionprops(label_image, intensity_image=img, coordinates='rc'):
        if region.major_axis_length == 0:
            continue
        else:
            region_ar = region.minor_axis_length / region.major_axis_length 
        if region.mean_intensity < 1:
            continue
        if region.area < min(area) or region.area > max(area):
            continue
        if region_ar < min(ar) or region_ar > max(ar):
            continue
        y, x = region.centroid
        major = region.major_axis_length
        minor = region.minor_axis_length
        angle = region.orientation / math.pi * 180
        if angle < 0:
            angle = -angle + 90
        else:
            angle = 90 - angle
        data.append([x, y, angle, major, minor, region.area])               
    column_names = ['X', 'Y', 'Angle', 'Major', 'Minor', 'Area']
    particles = pd.DataFrame(data=data, columns=column_names)
    return particles
def stack_track(imgStack, **kwargs):
    area = (250, 10000)
    ar = (0, 0.7)
    thres_percentile = 1
    for kw in kwargs:
        if kw == 'area':
            area = kwargs[kw]
        elif kw == 'ar':
            ar = kwargs[kw]
        elif kw == 'thres_percentile':
            thres_percentile = kwargs[kw]
    stack_particles = pd.DataFrame()
    for num, img in enumerate(imgStack):        
        particles = locate_particle(img, area=area, ar=ar, thres_percentile=thres_percentile).assign(Slice=num)
        stack_particles = stack_particles.append(particles)
        print(str(len(particles)) + ' particles are found in frame ' + str(num))    
    return stack_particles
    
if __name__ == '__main__':
    # Find particles
    folder = r'I:\Data\12032019\08'
    imgDirs = dirrec(folder, '*.tif')
    nameList = []
    dirList = []
    for imgDir in imgDirs:
        path, file = os.path.split(imgDir)
        name, ext = os.path.splitext(file)
        nameList.append(name)
        dirList.append(imgDir)
    fileList = pd.DataFrame()
    fileList = fileList.assign(Name=nameList, Dir=dirList)
    fileList = fileList.sort_values(by=['Name'])
    particles = pd.DataFrame()
    for num, imgDir in enumerate(fileList.Dir):  
        print('Frame %d' % num)
        img = io.imread(imgDir)
        particle = locate_particle(img, thres_percentile=.5, area=(300, 1200), ar=(0.5, 1))
        particle = particle.assign(frame=num)
        particles = particles.append(particle)
    particles.to_csv(os.path.join(folder, 'finding.csv'), index=False)