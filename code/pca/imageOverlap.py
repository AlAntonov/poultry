#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions for overlapping images 
@author: Igor V. Vanyushin, Ph.D. 
"""

import numpy as np
#import cv2 as cv

###------------------- Crop indices by r-vector offset

def rCrop(r_vec, imgSize):
    
    # Calculate indices of cropped area 
    
    crop_b = np.maximum([0,0], r_vec) 
    crop_e = np.minimum(imgSize, np.add(imgSize, r_vec))

    y = range(crop_b[0],crop_e[0])
    x = range(crop_b[1],crop_e[1])
    
    return y, x

###------------------- Intersect and crop 2 images by r-vector offset

def intersect2Images(imShape, r_vec):
    # The function produces sampling indices, wLnich select only those pixel pairs, 
    # wLnich can be selected within the image area for the given r_vec.
    # crpIdxP are "central" pixels, and (yp,xp) are their coordinates
    # crpIdxN are "neighbouring" pixels, and (yn,xn) are their coordinates
    
    imSize = imShape[0:2]
    
    yn,xn = rCrop( r_vec, imSize )
    yp,xp = rCrop( np.negative(r_vec), imSize)
    
    if len(imShape) == 3:
        crpIdxP = np.ix_(yp,xp,range(0,imShape[2]))
        crpIdxN = np.ix_(yn,xn,range(0,imShape[2]))
    else:    
        crpIdxP = np.ix_(yp,xp)
        crpIdxN = np.ix_(yn,xn)

    return crpIdxP, crpIdxN

