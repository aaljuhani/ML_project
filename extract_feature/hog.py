
from skimage import feature
import numpy as np
import cv2
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


class HOG:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self,image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns

        new_px = 500
        # we need to keep in mind aspect ratio so the image does
        # not look skewed or distorted -- therefore, we calculate
        # the ratio of the new image to the old image
        r = float(new_px) / image.shape[1]
        dim = (new_px, int(image.shape[0] * r))

        # perform the actual resizing of the image and show it
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        lbp = feature.local_binary_pattern(resized, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist