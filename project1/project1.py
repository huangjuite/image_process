

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import pandas as pd

img = plt.imread('Bird feeding 3 low contrast.tif')
# img = cv2.imread('Bird feeding 3 low contrast.tif')

cv2.imshow('bird', img)

min_v = math.atan((0-128)/32)
max_v = math.atan((255-128)/32)

print('minimum value:', min_v)
print('maximum value:', max_v)


def rescale(v):
    
    shift = (255-0)/2. - (max_v+min_v)/2
    scale = (255-0)/(max_v-min_v)

    num = v * scale + shift

    return int(round(num,0))

def transform(r) -> int:
    return rescale(math.atan((r-128)/32))

##########  Figure of s = T(r)  ############
x = np.arange(256)
y = np.array([transform(r) for r in x])

f = plt.figure()
plt.plot(x,y)
plt.title(r'T(r)')
plt.xlabel('r')
plt.ylabel('s')
plt.savefig('transform function')
plt.show(0)

##########  Table of transformation function  ############
content = np.vstack((x,y)).T
table = pd.DataFrame(content, columns = ['r', 's'])
table.to_csv('transform tabel.csv', index=False)

##########  transform image  ############
new_img = np.zeros(img.shape, dtype=np.uint8)
for i,k in np.ndindex(new_img.shape):
    new_img[i][k] = transform(img[i][k])

cv2.imshow('transformed', new_img)
cv2.imwrite('transformed.tif', new_img)
cv2.imwrite('transformed.png', new_img)


##########  histogram  ############
img_his = np.zeros([256], dtype=int)
transformed_his = np.zeros([256], dtype=int)

for i,k in np.ndindex(img.shape):
    img_his[img[i][k]] += 1

for i,k in np.ndindex(new_img.shape):
    transformed_his[new_img[i][k]] += 1

fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].bar(x,img_his)
axs[1].bar(x,transformed_his)
axs[0].set_title('original image histogram')
axs[1].set_title('transformed image histogram')
plt.savefig('histograms')
plt.show(0)


cv2.waitKey(0)