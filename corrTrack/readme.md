# corrTrack particle finding code
This code use the idea of image cross correlation to find particles in images. It works best when the particles to be found appear exactly the same. It requires an input image and a single particle mask to run. 
## Sample images
![sample images](https://github.com/ZLoverty/Python-GUI/blob/master/corrTrack/img/video.gif)
## Sample particle mask
![sample particle mask](https://github.com/ZLoverty/Python-GUI/blob/master/corrTrack/img/maski.png?raw=true)
## Comments
The white bright edges of the spherical particles are the most outstanding feature, so I choose a bright ring as the mask.
## How to use
In ``corrTrack.py``, edit the following lines:
```python
    img = io.imread('video.tif')
    mask = io.imread('maski.tif')
    num_images = img.shape[0]
    num_particles = 3
    nTotal = 100
```
Indicate the full directories of ``img`` and ``mask``. 
``num_particles`` is the total number of particles to be found in each frame.
``nTotal`` is the total number of frames to be analyzed.
