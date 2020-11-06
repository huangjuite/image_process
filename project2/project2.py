

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import pandas as pd

# read image
img = plt.imread('Bird 2.tif')
img = np.float32(img)

# dtft
ftimage = np.fft.fft2(img)
ftimage = np.fft.fftshift(ftimage)

# magnitude of dtft
magnitude = np.abs(ftimage)
scale = 255/np.log(np.max(magnitude))
magnitude = scale*np.log(magnitude)
magnitude = np.clip(np.uint8(magnitude),0,255)
cv2.imwrite('magnitude.tif', magnitude)


# ideal high, low pass filter
rows, cols = img.shape
crow, ccol = rows//2 , cols//2

mask_low = np.zeros((rows, cols), np.uint8)
mask_low[crow-30:crow+30, ccol-30:ccol+30] = 1

mask_high = np.ones((rows, cols), np.uint8)
mask_high[crow-30:crow+30, ccol-30:ccol+30] = 0

# low pass image
iftimage_low = ftimage*mask_low
iftimage_low = np.fft.ifftshift(iftimage_low)
iftimage_low = np.fft.ifft2(iftimage_low)
iftimage_low = np.uint8(np.abs(iftimage_low))
cv2.imwrite('iftimage_low.tif', iftimage_low)

# high pass image
iftimage_high = ftimage*mask_high
iftimage_high = np.fft.ifftshift(iftimage_high)
iftimage_high = np.fft.ifft2(iftimage_high)
iftimage_high = np.uint8(np.abs(iftimage_high))
cv2.imwrite('iftimage_high.tif', iftimage_high)

# top 25 frequency
magnitude = np.abs(ftimage)
mag_l, _ = np.hsplit(magnitude, 2)
mag = np.hstack((mag_l, np.zeros((512,256),dtype=np.uint8)))

for i in range(25):
    idx = np.unravel_index(mag.argmax(), mag.shape)
    v = mag[idx]
    mag[idx] = 0
    print("%02d, %.3f"%(i+1, v), idx)

scale = 255/np.log(np.max(mag)+1)
mag = scale*np.log(mag+1)
cv2.imwrite('magnitude_left.tif', mag)