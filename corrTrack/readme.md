# corrTrack particle finding code
This code use the idea of image cross correlation to find particles in images. It works best when the particles to be found appear exactly the same. It requires an input image and a single particle mask to run. 
## Sample images
![sample images](https://github.com/ZLoverty/Python-GUI/blob/master/corrTrack/img/video.gif)
## Sample particle mask
![sample particle mask](https://github.com/ZLoverty/Python-GUI/blob/master/corrTrack/img/maski.png?raw=true)
## Comments
The white bright edges of the spherical particles are the most outstanding feature, so I choose a bright ring as the mask.
