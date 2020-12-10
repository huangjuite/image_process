import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import pandas as pd


def rgb2hsi(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    H = np.zeros((R.shape))
    S = np.zeros((R.shape))
    I = np.zeros((R.shape))

    for ik in range(img.shape[0]):
        for ij in range(img.shape[1]):
            r = R[ik, ij]
            g = G[ik, ij]
            b = B[ik, ij]

            nom = ((r-g)+(r-b))/2
            denom = np.sqrt((r-g)**2 + (r-b)*(g-b))

            h = 0
            if denom == 0:
                h = 0
            elif b <= g:
                h = np.arccos(nom/denom)
            else:
                h = 2*np.pi - np.arccos(nom/denom)

            h = h/(2*np.pi)
            sumc = r+g+b
            if sumc == 0:
                s = 0
            else:
                s = 1-(3*min([r, g, b])/(r+g+b))
            i = (r+g+b)/3

            H[ik, ij] = h
            S[ik, ij] = s
            I[ik, ij] = i
    H = H.astype(np.float32)
    S = S.astype(np.float32)
    I = I.astype(np.float32)
    return H, S, I


def hsi2rgb(H, S, I):
    R = np.zeros(H.shape)
    G = np.zeros(H.shape)
    B = np.zeros(H.shape)

    for k in range(H.shape[0]):
        for j in range(H.shape[1]):
            h = H[k, j]
            s = H[k, j]
            i = H[k, j]
            r, g, b = i, i, i

            if s > 1e-6:
                h = h*360
                if h > 0 and h < 120:
                    b = i*(1-s)
                    r = i*(1 + (s*math.cos(math.radians(h)) /
                                math.cos(math.radians(60-h))))
                    g = 3*i-(r+b)
                elif h >= 240:
                    h = h-240
                    g = i*(1-s)
                    b = i*(1 + (s*math.cos(math.radians(h)) /
                                math.cos(math.radians(60-h))))
                    r = 3*i-(g+b)
                elif h > 120 and h <= 240:
                    h = h-120
                    r = i*(1-s)
                    g = i*(1 + (s*math.cos(math.radians(h)) /
                                math.cos(math.radians(60-h))))
                    b = 3*i-(g+r)
                R[k, j] = r
                G[k, j] = g
                B[k, j] = b

    return R, G, B


def laplacian_sharp(img, ker):
    m, n = ker.shape
    y, x = img.shape
    # padding
    pad_img = np.ones((y+2, x+2))
    pad_img[1:-1, 1:-1] = img

    new_img = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_img[i][j] = np.sum(pad_img[i:i+m, j:j+n] * ker)
    return new_img


# read degaded image
img = plt.imread('Bird 3 blurred.tif')
img = np.float32(img)/255.0
print(img.shape)

r = img[:, :, 0]
g = img[:, :, 1]
b = img[:, :, 2]
cv2.imwrite('r.png', (r*255))
cv2.imwrite('g.png', (g*255))
cv2.imwrite('b.png', (b*255))

hsiimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h = hsiimg[:, :, 0]/360.0
s = hsiimg[:, :, 1]
i = hsiimg[:, :, 2]

# h, s, i = rgb2hsi(img)

cv2.imwrite('h.png', (h*255))
cv2.imwrite('s.png', (s*255))
cv2.imwrite('i.png', (i*255))

laplacian_kernel = np.zeros((3, 3))
laplacian_kernel[0] = [-1, -1, -1]
laplacian_kernel[1] = [-1, 9, -1]
laplacian_kernel[2] = [-1, -1, -1]
nr = laplacian_sharp(r, laplacian_kernel)
ng = laplacian_sharp(g, laplacian_kernel)
nb = laplacian_sharp(b, laplacian_kernel)
sharp_rgb = np.dstack((nb, ng, nr))
cv2.imwrite('sharp_rgb.png', (sharp_rgb*255))

ni = laplacian_sharp(i, laplacian_kernel).astype(np.float32)

# hr, hg, hb = hsi2rgb(h, s, ni)
# sharp_hsi = np.dstack((hb, hg, hr))

sharp_hsi = cv2.cvtColor(np.dstack((h*360, s, ni)), cv2.COLOR_HSV2BGR)

cv2.imwrite('sharp_hsi.png', (sharp_hsi*255))
cv2.imwrite('difference.png', (sharp_rgb-sharp_hsi)*255)

