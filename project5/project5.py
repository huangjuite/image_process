import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import pandas as pd


def get_kernel(sigma):
    size = int(2*(np.ceil(3*sigma))+1)
    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * \
        np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal
    return kernel


def convolution(img, ker, padding=2):
    m, n = ker.shape
    y, x = img.shape
    # padding
    pad_img = np.ones((y+padding*2, x+padding*2))
    pad_img[padding:-padding, padding:-padding] = img

    new_img = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_img[i][j] = np.sum(pad_img[i:i+m, j:j+n] * ker)
    return new_img


def zero_crossing(img, padding=1, thres_hold=0):
    y, x = img.shape
    # padding
    pad_img = np.zeros((y+padding*2, x+padding*2))
    pad_img[padding:-padding, padding:-padding] = img
    thres = thres_hold*(np.max(pad_img)-np.min(pad_img))

    new_img = np.zeros((y, x))
    for i in range(y):
        for j in range(x):

            # left/right
            if pad_img[i-1][j]*pad_img[i+1][j] < 0 and \
                    (pad_img[i-1][j]-pad_img[i+1][j]) > thres:
                new_img[i][j] = 1
                continue

            # up/down
            if pad_img[i][j-1]*pad_img[i][j+1] < 0 and \
                    (pad_img[i][j-1]-pad_img[i][j+1]) > thres:
                new_img[i][j] = 1
                continue

            # left diagonals
            if pad_img[i-1][j-1]*pad_img[i+1][j+1] < 0 and \
                    (pad_img[i-1][j-1]-pad_img[i+1][j+1]) > thres:
                new_img[i][j] = 1
                continue

            # right diagonals
            if pad_img[i-1][j+1]*pad_img[i-1][j+1] < 0 and \
                    (pad_img[i-1][j+1]*pad_img[i-1][j+1]) > thres:
                new_img[i][j] = 1
                continue

    return new_img


# read image
img = plt.imread('Car On Mountain Road.tif')
img = np.float32(img)/255.0
print(img.shape)

log_kernel = get_kernel(sigma=3)

LoG = convolution(img, log_kernel, padding=log_kernel.shape[0]//2)
LoG_norm = (LoG-np.min(LoG))/(np.max(LoG)-np.min(LoG))
cv2.imwrite('LoG.png', (LoG_norm*255))

log_0_percent = zero_crossing(LoG, thres_hold=0)
cv2.imwrite('log_0_percent.png', (log_0_percent*255))

log_4_percent = zero_crossing(LoG, thres_hold=0.04)
cv2.imwrite('log_4_percent.png', (log_4_percent*255))
