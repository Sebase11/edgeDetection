import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from matplotlib.pyplot import imread

inputImg = imread("py\cat.jpg")

gamma = 1.04

r = inputImg[:,:,0]
g = inputImg[:,:,1]
b = inputImg[:,:,2]

rConst = 0.2128
gConst = 0.7152
bConst = 0.0722

grayScaleImg = rConst * r ** gamma + gConst * g ** gamma + bConst * b ** gamma

gaussianBlureKernal = np.array([[1, 4, 6, 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]], dtype=np.float32)

edgeDetectionKernal = np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=np.float32)

gaussianBlureKernal_sum = np.sum(gaussianBlureKernal)
if gaussianBlureKernal_sum != 0:  
    gaussianBlureKernal = gaussianBlureKernal / gaussianBlureKernal_sum


gaussianBlureImg = convolve2d(grayScaleImg, gaussianBlureKernal, mode='same', boundary='fill', fillvalue=0)

edgeDetectionImg = convolve2d(gaussianBlureImg, edgeDetectionKernal, mode="same", boundary='fill', fillvalue=0)

edgeDetectionImg = np.clip(edgeDetectionImg, 0, 255)

binedgeDetectionImg = np.where(edgeDetectionImg > 10, 255, 0)





fig = plt.figure(1)
img1, img2 = fig.add_subplot(121), fig.add_subplot(122)

img1.imshow(inputImg)
img2.imshow(binedgeDetectionImg, cmap=plt.cm.get_cmap("gray"))

fig.show()
plt.show()