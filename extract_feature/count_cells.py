
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure
import mahotas as mh
from pylab import imshow, show

img = cv2.imread('Small_test_images_jpeg/test_tiny.jpg',0)

######################Otsu_thresholding_option1######################
val = filters.threshold_otsu(img)
hist, bins_center = exposure.histogram(img)
plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.title('Original Noisy Image')
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(132)
plt.title('Otsu thresholding')
plt.imshow(img < val, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.title('Histogram')
plt.plot(bins_center, hist, lw=2)
plt.axvline(val, color='k', ls='--')
plt.tight_layout()
plt.show()
##################################################################
########################Counting cells with connected component labeling with otsu option1########################
labeled, nr_objects = mh.label(img < val)
print("Number of objects found: "+ str(nr_objects))
imshow(labeled)
show()
#plt.imshow(labeled)

#plt.jet()

##################################################################
######################Otsu_thresholding_option2############################################
ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure(figsize=(9, 4))
plt.subplot(131), plt.imshow(img,cmap = 'gray')
plt.title('Original Noisy Image')#, plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(imgf,cmap = 'gray')
plt.title('Otsu thresholding')#, plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.hist(img.ravel(), 256)
plt.axvline(x=ret, color='r', linestyle='dashed', linewidth=2)
plt.title('Histogram')#, plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
##################################################################
########################Counting cells with connected component labeling otsu option2########################
labeled, nr_objects = mh.label(imgf)
print("Number of objects found: "+ str(nr_objects))
imshow(labeled)
show()
#plt.imshow(labeled)

#plt.jet()

