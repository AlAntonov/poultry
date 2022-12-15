#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 8 2022
@author: Igor Vanyushin, Ph.D.
referenced to the following source:
https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 
from ioImageData import loadFramesDir, saveImage2sRGB
from sklearn.decomposition import PCA

# Input models for eigenchickens:
inputDir_model = "./Data/input/Model 2/" 
modelIn_name = "*.*"
maxMdlNum = 8 # Maximal number of model images to be loaded for eigenchickens

# Input samplpes for chickens detection:
inputDir_sample = "./Data/input/"
sampleIn_name = "Poultry_test.bmp"
maxSmpNum = 1 # Maximal number of samlpe images to be loaded for detection

# Main paths:
pathModel = inputDir_model + modelIn_name
pathSample = inputDir_sample + sampleIn_name

#------------------- Load sample images for PCA analysis:
models, BPP = loadFramesDir( pathModel, maxMdlNum, colOut = 'xyz' ) # Load images for PCA
model = models[0] #.astype(np.float64)
model_shape = model.shape

#------------------- Convert model images into a single flatten matrix:
chick_matrix = []

for mdl in models: # Load models for PCA
    mdl_rot = mdl[:,:,1]  # Only Y-channel is used
    for rot in range(0,4): # Additional rotated images for more of orientations:
        chick_matrix.append(mdl_rot.flatten()) # Flatten to a single-dimentional vector
        mdl_rot = cv2.rotate(mdl_rot,cv2.ROTATE_90_CLOCKWISE) # Rotate 90 deg. to create a new vector
 
chick_matrix = np.array(chick_matrix)

#================================== PCA fitting:
pca = PCA().fit(chick_matrix)
print(pca.explained_variance_ratio_) # Show strength of each component

# First n principal components will be used as eigenchicks:
eigchick_num = 4 # This number can be automated using pca.explained_variance_ratio
eigenchickens = pca.components_[:eigchick_num]

# Generate weights as a KxN matrix: K is the number of eigenchicks and N the number of samples
weights = eigenchickens @ (chick_matrix - pca.mean_).T

#================================== Test on sample images
#------------------- Load samples for recognition:
    
samples, BPP = loadFramesDir( pathSample, maxSmpNum, colOut = 'xyz' )
sample = samples[0] 
sample_shape = sample.shape
msz = model_shape

Ych = sample[:,:,1] # Only Y-channel will be used for PCA recognition
output_map = np.zeros(Ych.shape)

est_level = 0.1  # Matching accuracy
    
# Scanning big image with a set of eigenchicks:
for y in range(0, sample_shape[0]-model_shape[0],1):
    for x in range(0, sample_shape[1]-model_shape[1],1):

        # Getting quiery at each aperture position:
        smp = Ych[y:(y+msz[0]),x:(x+msz[1])] # Getting a sample from the big image
        query = smp.reshape(1,-1) # 2D sample to 1-dimentional vector
        query_weight = eigenchickens @ (query - pca.mean_).T # weights of query related to eigenchicks
        euclidean_distance = np.linalg.norm(weights - query_weight, axis=0) # Matching through Euclidean distance
        best_match = np.argmin(euclidean_distance) # The number of the best match
        ed_est = euclidean_distance[best_match]/model_shape[0]/model_shape[1] # The norm should be verified!
        
        if ed_est < est_level: # Check for acceptable matching and showing on the map:
            ch_mx = chick_matrix[best_match].reshape(model_shape[0:2]) # Show the best match on the map
            output_map[y:(y+msz[0]),x:(x+msz[1])] = np.maximum(output_map[y:(y+msz[0]),x:(x+msz[1])], ch_mx)


#------------------- Show/save results:

imgOut = np.zeros(sample_shape)
imgOut = sample/2
imgOut[:,:,2] = output_map
#imgOut = (imgOut + samplePyr[pyrLev]*50)/2
# img2save = cv2.cvtColor(np.float32(imgOut), cv2.COLOR_RGB2BGR)  
# cv2.imwrite( 'Poultry_test.bmp', img2save )  

I0 = 255
img_out = imgOut/I0
imgOut_name = 'Poultry_test_out.bmp'
saveImage2sRGB(imgOut_name, img_out, inType='xyz')
 
   

