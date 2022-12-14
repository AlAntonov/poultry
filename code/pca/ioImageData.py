#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue August 16 2022

@author: Igor V. Vanyushin, Ph.D. 
"""

# Libraries required:
import numpy as np
import cv2
import glob
  
###----------------------------------------------------------------------------
###--------------------- Color Conversion section -----------------------------

### Color constants
sRGB2XYZ_Mx = [
      [  0.4124564,  0.3575761,  0.1804375 ],
      [  0.2126729,  0.7151522,  0.0721750 ],
      [  0.0193339,  0.1191920,  0.9503041 ]  ]

###------ Color conversion sRGB -> XYZ (8-bit images only by sRGB standard!)

def srgb2xyz(im_srgb, BPP):
    
    I0 = 2**BPP - 1
    
    im_srgb = np.array(im_srgb)/I0
    im_rgb_lin = np.copy(im_srgb) # Initialize
    H, W, cn = np.shape(im_srgb) 

    # Linearization:
    rng_lo = im_srgb <= 0.04045
    rng_hi = ~rng_lo
    im_rgb_lin[ rng_lo ] = im_srgb[rng_lo]/12.92
    im_rgb_lin[ rng_hi ] = ((im_srgb[ rng_hi ] + 0.055)/(1.0 + 0.055)) ** 2.4
            
    # Linear transform:    
    convMx = np.array(sRGB2XYZ_Mx)
    
    im_out = np.dot(im_rgb_lin, convMx.T)        

    return I0 * np.clip(im_out, 0, 1)


###------ Color conversion  XYZ->sRGB (8-bit images only by sRGB standard!)

def xyz2srgb(im_xyz):
                
    # Linear color transformation:   
    XYZ2sRGB_MX = np.linalg.inv(sRGB2XYZ_Mx)
    
    RGB_lin = np.dot(im_xyz, XYZ2sRGB_MX.T)
    
    # Apply sRGB gamma:
    lin_mask = RGB_lin <= 0.0031308
    im_out = np.zeros(RGB_lin.shape)
    im_out[lin_mask] = 12.92 * RGB_lin[lin_mask]
    im_out[~lin_mask] = 1.055 * RGB_lin[~lin_mask]**(1.0/2.4) - 0.055                   
    
    return np.clip(im_out, 0, 1)

def xyz2logYcc(image, BPP):
    
    imLog = np.log2( np.maximum(2**-BPP, image) ) - BPP
    imLog[:,:,0] = imLog[:,:,0] - imLog[:,:,1]
    imLog[:,:,2] = imLog[:,:,2] - imLog[:,:,1]
    
    return imLog 

def logYcc2xyz(imLogYcc, BPP):
    
    imLogXYZ = np.copy(imLogYcc)
    imLogXYZ[:,:,0] = imLogYcc[:,:,0] + imLogYcc[:,:,1]
    imLogXYZ[:,:,2] = imLogYcc[:,:,2] + imLogYcc[:,:,1]
    imXYZ = 2**(imLogXYZ + BPP)
    
    return imXYZ 

###----------------------------------------------------------------------------
### ------------------------ Image Loading functions --------------------------
 
###------ Load a single image in XYZ format

def loadImage2RGB(imageName):
    image = cv2.imread(imageName, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    BPP = 8 * image.dtype.itemsize
    imageRGB = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
                         
    return imageRGB, BPP
                         

def loadImage2XYZ(imageName):
    image, BPP = loadImage2RGB(imageName)
    imageXYZ = srgb2xyz(image, BPP)
    
    return imageXYZ, BPP

###------ Save a single image from XYZ to sRGB format

def saveImage2sRGB(imageName, image, inType= 'rgb'):
    
    if inType == 'xyz':
        image = 255*xyz2srgb(image)
        
    img2save = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2BGR)  
    
    return cv2.imwrite( imageName, img2save )

###----------------------------------------------------------------------------
###------ Load a set of images

def loadFramesDir(path, maxFrames, colOut = 'rgb'):

    #listOfImages = []
    frames = []
    
    # Progress bar init:
    print('Loading frames/images:')
    
    fln = 0
    
    for fname in glob.iglob(path, recursive=True):
    
        print(fname)
    #    listOfImages.append(fname)
        if colOut == 'rgb':
            image, BPP = loadImage2RGB(fname)
        elif colOut == 'xyz':  
            image, BPP = loadImage2XYZ(fname)
        else: 
            print("Unknown output color space")
            break
        
        frames.append(image)     
        
        fln = fln + 1     
        if fln == maxFrames: break
    
    print('completed')    
    return frames, BPP


