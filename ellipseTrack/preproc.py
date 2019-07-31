from skimage import io, util
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import os
import pdb


def threshold(img, percentile=1):
    threshold = np.percentile(img, percentile)
    imgt = img.copy()
    imgt[img<=threshold] = 255
    imgt[img>threshold] = 0
    return imgt
def stack_thres(imgStack, percentile=1):    
    thresholded = []
    for img in imgStack:    
        imgt = threshold(img, percentile=percentile)
        thresholded.append(imgt)
    return np.array(thresholded)
def locate_particle(img, **kwargs):
    area = (250, 1000)
    ar = (0, 0.7)
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
def particle_on_img(img, particles):
    plt.imshow(img, cmap='gray')
    ax = plt.gca()
    for num, particle in particles.iterrows():
        x = particle.X
        y = particle.Y
        angle = particle.Angle        
        major = particle.Major
        minor = particle.Minor
        elli = mpatches.Ellipse((x, y), major, minor, angle, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(elli)
    plt.show()
def stack_track(imgStack, **kwargs):
    area = (250, 1000)
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
def preproc(imgDir):
    """
    Input
        A full directory of a bandpass filtered image.
        Image should be a tiff stack.
    Process
        1 ... Threshold filter the image (with custom percentile threshold)
        2 ... Thresholded images are saved to the same directory of the original
                image, named "thresholded.tif".
        3 ... Find particles in thresholded images (with custom parameter ranges 
                such as area and aspect ratio
        4 ... Particle finding results are saved in a *.csv file, named "finding.csv".
    """
    folder, file = os.path.split(imgDir)
    if os.path.exists(os.path.join(folder, 'finding.csv')) and \
        os.path.exists(os.path.join(folder, 'thresholded.tif')):
        print('Images in ' + folder + ' has already been preprocessed.')
        return
    print('Loading images from ' + imgDir + ' ...')
    imgStack = io.imread(imgDir)
    print('Particle finding begins ...')
    particles = stack_track(imgStack)    
    particles.to_csv(os.path.join(folder, 'finding.csv'))
    print('Saving thresholded images ...')
    thresholded = stack_thres(imgStack, percentile=1) 
    io.imsave(os.path.join(folder, 'thresholded.tif'), thresholded)
    
if __name__ == '__main__':
    imgDir = r'F:\Data(2019)\07142019\07\processed_img.tif'
    # preproc(imgDir)
    pdb.set_trace()