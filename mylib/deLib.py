"""
GENERAL
=======
Double emulsion project code library.
"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage import io
from myImageLib import readdata
from pivLib import read_piv, PIV
import corrTrack
import os
# %% codecell

class droplet_image:
    """Container of functions related to confocal droplet images"""
    def __init__(self, image_sequence):
        """image_sequence: dataframe of image dir info, readdata return value
        mask: image mask, the same as PIV mask
        xy0: 2-tuple of droplet initial coordinates, read from positions.csv file
        mask_shape: shape of the circular mask, 2-tuple specifying the rect bounding box of the mask, typically a square
                    read from positions.csv file"""
        self.sequence = image_sequence
    def __len__(self):
        return len(self.sequence)
    def process_first_image(self, mask):
        """Compare the provided mask (template) with the first image.
        If the maximum correlation is at the center of the image, the input mask should be correct.
        This function also records the measured center coordinates (xc, yc), for computing the offset."""
        img = io.imread(self.sequence.Dir[0])
        h, w = img.shape
        xy, pkv = corrTrack.track_spheres(img, mask, 1, subpixel=True)
        xyc = np.flip(xy.squeeze())
        # The center coords should be very close to the image center (w/2, h/2)
        diff = np.square(xyc-np.array([w/2, h/2])).sum()
        if diff > 10: # this threshold is arbitrary
            print("The detected center is {:.1f} pixels from the image center, check if the input mask is correct.".format(diff**0.5))
        return xyc
    def droplet_traj(self, mask, xy0):
        xyc = self.process_first_image(mask)
        xym_list = []
        for num, i in self.sequence.iterrows():
            img = io.imread(i.Dir)
            xy, pkv = corrTrack.track_spheres(img, mask, 1)
            xym_list.append(np.flip(xy.squeeze()))
        xym = np.stack(xym_list, axis=0)
        traj = self.sequence.copy()
        traj = traj.assign(x=xy0[0] + xym[:, 0] - xyc[0], y=xy0[1] + xym[:, 1] - xyc[1])
        self.traj = traj
        return traj
    def get_cropped_image(self, index, traj, mask_shape):
        """Retrieve cropped image by index"""
        x = int(traj.x[index])
        y = int(traj.y[index])
        h, w = mask_shape
        x0 = x - w // 2
        x1 = x0 + w
        y0 = y - h // 2
        y1 = y0 + h
        img = self.get_image(index)
        cropped = img[y0:y1, x0:x1]
        return cropped
    def get_image(self, index):
        """Retrieve image by index"""
        imgDir = self.sequence.Dir[index]
        img = io.imread(imgDir)
        return img
    def check_traj(self, traj, mask_shape, n=10):
        """Check the droplet finding.
        traj: DataFrame containing columns 'x' and 'y'
        mask_shape: 2-tuple, (width, height)
        n: total number of frames to inspect, default to 10"""
        interval = len(traj) // n
        count = 0
        for num, i in traj[::interval].iterrows():
            fig, ax = plt.subplots()
            img = io.imread(i.Dir)
            elli = Ellipse((i.x, i.y), *mask_shape, facecolor=(0,0,0,0), lw=1, edgecolor="red")
            ax.imshow(img, cmap="gray")
            ax.add_patch(elli)
            count += 1
            if count >= n:
                break
    def test(self):
        image_sequence = readdata(os.path.join(folder, "16", "raw"), "tif")[:1000]
        mask = io.imread(os.path.join(folder, "mask", "16.tif"))
        xy0 = (173, 165)
        mask_shape = (174, 174)
        di = droplet_image(image_sequence, mask, xy0, mask_shape)
    def get_image_name(self, index):
        return self.sequence.Name[index]
    def save_params(self, save_folder):
        params = {"mask_shape": (int(self.mask_shape[0]), int(self.mask_shape[1]))}
        with open(os.path.join(save_folder, "piv_params.json"), "w") as f:
            json.dump(params, f)
    def moving_mask_piv(self, save_folder, winsize, overlap, dt, mask_dir, xy0, mask_shape):
        """Perform moving mask PIV and save PIV results and parameters in save_folder"""
        mask = io.imread(mask_dir)
        if os.path.exists(save_folder) == False:
            os.makedirs(save_folder)
        traj = self.droplet_traj(mask, xy0)
        for i0, i1 in zip(self.sequence.index[::2], self.sequence.index[1::2]):
            I0 = self.get_cropped_image(i0, traj, mask_shape)
            I1 = self.get_cropped_image(i1, traj, mask_shape)
            x, y, u, v = PIV(I0, I1, winsize, overlap, dt)
            # generate dataframe and save to file
            data = pd.DataFrame({"x": x.flatten(), "y": y.flatten(), "u": u.flatten(), "v": v.flatten()})
            data.to_csv(os.path.join(save_folder, "{0}-{1}.csv".format(self.get_image_name(i0), self.get_image_name(i1))), index=False)
        params = {"winsize": winsize,
                  "overlap": overlap,
                  "dt": dt,
                  "mask_dir": mask_dir,
                  "droplet_initial_position (xy0)": (int(xy0[0]), int(xy0[1])),
                  "mask_shape": (int(mask_shape[0]), int(mask_shape[1]))}
        # save traj and param data in .json files, so that only PIV data are saved in .csv files
        traj.to_json(os.path.join(save_folder, "droplet_traj.json"))
        with open(os.path.join(save_folder, "piv_params.json"), "w") as f:
            json.dump(params, f)
    def piv_overlay(self, piv_folder, out_folder):
        pass
    def piv_overlay_moving(self, piv_folder, out_folder, traj, piv_params, sparcity=1):
        """Draw PIV overlay for moving mask piv data (only on cropped images)"""
        def determine_arrow_scale(u, v, sparcity):
            row, col = u.shape
            return max(np.nanmax(u), np.nanmax(v)) * col / sparcity / 1.5
        if os.path.exists(out_folder) == False:
            os.makedirs(out_folder)
        mask_shape = piv_params["mask_shape"]
        l = readdata(piv_folder, "csv")
        # determine scale using the first frame
        x, y, u, v = read_piv(l.Dir[0])
        scale = determine_arrow_scale(u, v, sparcity)

        for num, i in l.iterrows():
            name = i.Name.split("-")[0]
            index = self.sequence.loc[self.sequence.Name==name].index[0]
            img = self.get_cropped_image(index, traj, mask_shape)
            x, y, u, v = read_piv(i.Dir)
            # sparcify
            row, col = x.shape
            xs = x[0:row:sparcity, 0:col:sparcity]
            ys = y[0:row:sparcity, 0:col:sparcity]
            us = u[0:row:sparcity, 0:col:sparcity]
            vs = v[0:row:sparcity, 0:col:sparcity]
            # plot quiver
            dpi = 300
            figscale = 1
            w, h = img.shape[1] / dpi, img.shape[0] / dpi
            fig = Figure(figsize=(w*figscale, h*figscale)) # on some server `plt` is not supported
            canvas = FigureCanvas(fig) # necessary?
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img, cmap='gray')
            ax.quiver(xs, ys, us, vs, color='yellow', width=0.003, \
                        scale=scale, scale_units='width') # it's better to set a fixed scale, see *Analysis of Collective Motions in Droplets* Section IV.A.2 for more info.
            ax.axis('off')
            # save figure
            fig.savefig(os.path.join(out_folder, name + '.jpg'), dpi=dpi)


# %% codecell
class fixed_mask_PIV:
    def PIV_masked_1(self, I0, I1, winsize, overlap, dt, mask):
        """Test different masking procedures.
        1. Mask raw images by setting background 0, then apply PIV.
        2. Mask raw images by setting background nan, then apply PIV.
        3. Apply PIV directly on raw images, then apply mask on velocity.
        """
        assert(mask.shape==I0.shape)
        mask = mask >= mask.mean() # convert mask to boolean array
        I0 = I0 * mask
        I1 = I1 * mask
        x, y, u, v = PIV(I0, I1, winsize, overlap, dt)
        mask_w = divide_windows(mask, windowsize=[winsize, winsize], step=winsize-overlap)[2] >= 1
        assert(mask_w.shape==x.shape)
        u[~mask_w] = np.nan
        v[~mask_w] = np.nan
        return x, y, u, v
    def PIV_masked_2(self, I0, I1, winsize, overlap, dt, mask):
        """Test different masking procedures.
        1. Mask raw images by setting background 0, then apply PIV.
        2. Mask raw images by setting background nan, then apply PIV.
        3. Apply PIV directly on raw images, then apply mask on velocity.
        """
        assert(mask.shape==I0.shape)
        mask = mask >= mask.mean() # convert mask to boolean array
        x, y, u, v = PIV(I0, I1, winsize, overlap, dt)
        mask_w = divide_windows(mask, windowsize=[winsize, winsize], step=winsize-overlap)[2] >= 1
        assert(mask_w.shape==x.shape)
        u[~mask_w] = np.nan
        v[~mask_w] = np.nan
        return x, y, u, v
    def PIV_masked_3(self, I0, I1, winsize, overlap, dt, mask):
        """Test different masking procedures.
        1. Mask raw images by setting background 0, then apply PIV.
        2. Mask raw images by setting background nan, then apply PIV.
        3. Apply PIV directly on raw images, then apply mask on velocity.
        """
        assert(mask.shape==I0.shape)
        mask = mask >= mask.mean() # convert mask to boolean array
        I0[~mask] = np.nan
        I1[~mask] = np.nan
        x, y, u, v = PIV(I0, I1, winsize, overlap, dt)
        mask_w = divide_windows(mask, windowsize=[winsize, winsize], step=winsize-overlap)[2] >= 1
        assert(mask_w.shape==x.shape)
        u[~mask_w] = np.nan
        v[~mask_w] = np.nan
        return x, y, u, v
    def test(self):
        I0 = io.imread(os.path.join("img", "I10.tif"))
        I1 = io.imread(os.path.join("img", "I11.tif"))
        mask = io.imread(os.path.join("img", "mask1.tif"))
        winsize = 40
        overlap = 20
        dt = 0.02
        x, y, u, v = [], [], [], []
        for func in [PIV_masked_1, PIV_masked_2]:
            x1, y1, u1, v1 = func(I0, I1, winsize, overlap, dt, mask)
            x.append(x1); y.append(y1); u.append(u1); v.append(v1)
        fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200)
        for i in [0, 1]:
            ax[i].imshow(I0, cmap='gray')
            ax[i].quiver(x[i], y[i], u[i], v[i], color='yellow')
            ax[i].axis('off')
        print("The two masking procedures don't produce very different results, according to visual inspection.")
        print("I also plot the velocity distribution function below.")
        fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200)
        for i in [0, 1]:
            hist, bin_edges = np.histogram(u[i][~np.isnan(u[i])], density=True)
            ax[0].plot(bin_edges[:-1], hist)
            hist, bin_edges = np.histogram(v[i][~np.isnan(v[i])], density=True)
            ax[1].plot(bin_edges[:-1], hist)
        ax[0].set_xlabel("u")
        ax[1].set_xlabel("v")
        ax[0].set_ylabel("PDF")
        print("The two methods show statistically very similar results")
        print("Test the speed of the two methods.")
        t = []
        t.append(time.monotonic())
        for func in [PIV_masked_1, PIV_masked_2]:
            x1, y1, u1, v1 = func(I0, I1, winsize, overlap, dt, mask)
            t.append(time.monotonic())
        t1 = t[1] - t[0]
        t2 = t[2] - t[1]
        plt.bar([1, 2], [t1, t2])
        plt.xticks([1, 2])
        plt.ylabel("time (s)")
        print("The second method takes longer time. So the first method is better.")
        print("Since it gives similar results while using less time.")
        I20 = io.imread(os.path.join("img", "I20.tif"))
        I21 = io.imread(os.path.join("img", "I21.tif"))
        mask2 = io.imread(os.path.join("img", "mask2.tif"))
        x, y, u, v = PIV_masked(I20, I21, 20, 10, 0.02, mask2)
        fig, ax = plt.subplots()
        ax.imshow(I20, cmap='gray')
        ax.quiver(x, y, u, v, color='yellow')
        ax.axis('off')

if __name__=="__main__":
    # %% codecell
    # Test droplet_image class
    # %% codecell
    # create object
    folder = r"test_images\moving_mask_piv\raw"
    l = readdata(folder, "tif")
    DI = droplet_image(l)
    # %% codecell
    # test droplet_traj()
    mask_dir = r"test_images\moving_mask_piv\mask.tif"
    mask = io.imread(mask_dir)
    xy0 = (178, 161)
    traj = DI.droplet_traj(mask, xy0)
    traj
    # %% codecell
    # test check_traj()
    mask_shape = (174, 174)
    DI.check_traj(traj, mask_shape, n=8)
    # %% codecell
    img = DI.get_image(0)
    plt.imshow(img)
    # %% codecell
    mask_shape = (174, 174)
    img = DI.get_cropped_image(0, traj, mask_shape)
    plt.imshow(img)
    # %% codecell
    save_folder = r"test_images\moving_mask_piv\piv_result"
    winsize = 20
    overlap = 10
    dt = 0.02
    mask_dir = r"test_images\moving_mask_piv\mask.tif"
    xy0 = (178, 161)
    mask_shape = (174, 174)
    DI.moving_mask_piv(save_folder, winsize, overlap, dt, mask_dir, xy0, mask_shape)
    # %% codecell
    pd.read_json(os.path.join(save_folder, "droplet_traj.json"))
    # %% codecell
    piv_folder = r"test_images\moving_mask_piv\piv_result"
    out_folder = r"test_images\moving_mask_piv\piv_overlay_moving"
    traj = pd.read_json(os.path.join(piv_folder, "droplet_traj.json"))
    with open(os.path.join(piv_folder, "piv_params.json"), "r") as f:
        piv_params = json.load(f)
    DI.piv_overlay_moving(piv_folder, out_folder, traj, piv_params, sparcity=1)
    # %% codecell
    out_folder = r"test_images\moving_mask_piv\cropped_images"
    if os.path.exists(out_folder) == False:
        os.makedirs(out_folder)
    dpi = 300
    figscale = 1
    for i in DI.sequence.index:
        img = DI.get_cropped_image(i, traj, (174, 174))
        w, h = img.shape[1] / dpi, img.shape[0] / dpi
        fig = Figure(figsize=(w*figscale, h*figscale)) # on some server `plt` is not supported
        canvas = FigureCanvas(fig) # necessary?
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, cmap='gray')
        ax.axis("off")
        fig.savefig(os.path.join(out_folder, "{}.jpg".format(DI.get_image_name(i))))
