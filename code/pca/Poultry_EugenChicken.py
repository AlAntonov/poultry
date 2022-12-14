#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 5 2022
@author: Igor Vanyushin, Ph.D.
referenced to the following source:
https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 
from ioImageData import loadFramesDir
from sklearn.decomposition import PCA

# Input models for eigenchickens:
inputDir_model = "./Data/input/Model 1/" 
modelIn_name = "*.*"
maxMdlNum = 8 # Maximal number of model images to be loaded for eigenchickens

# Input samplpes for chickens detection:
inputDir_sample = "./Data/input/Samples 1/"
sampleIn_name = "*.*"
maxSmpNum = 8 # Maximal number of samlpe images to be loaded for detection

# Main paths:
pathModel = inputDir_model + modelIn_name
pathSample = inputDir_sample + sampleIn_name

#------------------- Load sample images for PCA analysis:
models, BPP = loadFramesDir( pathModel, maxMdlNum, colOut = 'xyz' ) # Load images for PCA
model = models[0] 
model_shape = model.shape

#------------------- Convert model images into a single flatten matrix:
chick_matrix = []

for mdl in models:
    mdl_rot = mdl[:,:,1]  # Only Y-channel is used
    for rot in range(0,4): # Additional rotated images for more of orientations:
        chick_matrix.append(mdl_rot.flatten()) 
        mdl_rot = cv2.rotate(mdl_rot,cv2.ROTATE_90_CLOCKWISE)
 
chick_matrix = np.array(chick_matrix)

#================================== PCA fitting:
pca = PCA().fit(chick_matrix)
print(pca.explained_variance_ratio_) # Show strength of each component

# First n principal components will be used as eigenchicks:
eigchick_num = 4 # This number can be automated using pca.explained_variance_ratio
eigenchickens = pca.components_[:eigchick_num]
 
# Show the first 4 eigenchicks as images:
fig, axes = plt.subplots(2,np.int8(eigchick_num/2),sharex=True,sharey=True,figsize=(8,8))
for i in range(eigchick_num):
    axes[i%2][i//2].imshow(eigenchickens[i].reshape(model_shape[0:2]), cmap="gray")
plt.show()

# Generate weights as a KxN matrix: K is the number of eigenchicks and N the number of samples
weights = eigenchickens @ (chick_matrix - pca.mean_).T

#================================== Test on sample images
#------------------- Load samples:
samples, BPP = loadFramesDir( pathSample, maxSmpNum, colOut = 'xyz' )
sample = samples[7] #.astype(np.float64)
sample_shape = sample.shape

#------------------- Calculate query weights for chicken recognition:
query = sample[:,:,1].reshape(1,-1) # Only Y-channel is used
query_weight = eigenchickens @ (query - pca.mean_).T
euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
best_match = np.argmin(euclidean_distance) # Finding the best match number 

# Visualize
fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
axes[0].imshow(query.reshape(model_shape[0:2]), cmap="gray")
axes[0].set_title("Query")
axes[1].imshow(chick_matrix[best_match].reshape(model_shape[0:2]), cmap="gray")
axes[1].set_title("Best match")
plt.show()


