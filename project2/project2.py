import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import pandas as pd


def get_norm_magnitude(ft):
    mg = np.abs(ft)
    scale = 255/np.log(np.max(mg))
    mg = scale*np.log(mg)
    mg = np.clip(np.uint8(mg), 0, 255)
    return mg


# read image
img = plt.imread('Bird 2.tif')
img = np.float32(img)

# dtft
ftimage = np.fft.fft2(img)
ftimage = np.fft.fftshift(ftimage)

# magnitude of dtft
magnitude = get_norm_magnitude(ftimage)
cv2.imwrite('magnitude.png', magnitude)


# ideal high, low pass filter
rows, cols = img.shape
crow, ccol = rows//2, cols//2

mask_low = np.zeros((rows, cols), np.uint8)
for x in range(rows):
    for y in range(cols):
        dist = np.linalg.norm(np.array([x, y]) - np.array([crow, ccol]))
        if dist < 30:
            mask_low[x, y] = 1

mask_high = 1 - mask_low

cv2.imwrite('low_pass_mask.png', mask_low*255)
cv2.imwrite('high_pass_mask.png', mask_high*255)

# low pass image
ftimage_low_mask = ftimage*mask_low
iftimage_low = np.fft.ifftshift(ftimage_low_mask)
iftimage_low = np.fft.ifft2(iftimage_low)
iftimage_low = np.clip(np.uint8(np.abs(iftimage_low)), 0, 255)

ftimage_low_mask = get_norm_magnitude(ftimage_low_mask)
cv2.imwrite('ftimage_low.png', ftimage_low_mask)
cv2.imwrite('iftimage_low.png', iftimage_low)

# high pass image
ftimage_high_mask = ftimage*mask_high
iftimage_high = np.fft.ifftshift(ftimage_high_mask)
iftimage_high = np.fft.ifft2(iftimage_high)
iftimage_high = np.clip(np.uint8(np.abs(iftimage_high)), 0, 255)

ftimage_high_mask = get_norm_magnitude(ftimage_high_mask)
cv2.imwrite('ftimage_high.png', ftimage_high_mask)
cv2.imwrite('iftimage_high.png', iftimage_high)

# top 25 frequency
magnitude = np.abs(ftimage)
mag_l, _ = np.hsplit(magnitude, 2)
mag = np.hstack((mag_l, np.zeros((512, 256), dtype=np.uint8)))
leftmag = get_norm_magnitude(mag)
cv2.imwrite('magnitude_left.png', leftmag)

print('\n #|    magnitude| (u, v)')
print('============================')
for i in range(25):
    idx = np.unravel_index(mag.argmax(), mag.shape)
    v = mag[idx]
    mag[idx] = 0
    print("%02d|"%(i+1), "%12.3f|"%(v), idx)
print('============================')

