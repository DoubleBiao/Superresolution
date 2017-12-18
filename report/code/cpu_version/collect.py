#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:08:10 2017

@author: qcat
"""
import numpy as np
from scipy.signal import convolve2d as conv2d

def collect(images, scale, filters, window, overlap, border):
    """
    :type images:   List[ndarray]
    :type scale:    int
    :type filters:  dict
    :type window:   List[int]
    :type overlap:  List[int]
    :type border:   List[int]    
    """

    # Initialization of features and total number of features 
    num_of_imgs = len(images)
    feature_cell = {}
    num_of_features = 0
    
    feature_size = -1

    # Extraction of features for each input image    
    for i in range(num_of_imgs): 

        # Extract features
        F = extract(images[i], scale, filters, window, overlap, border)

        # Calculate total number of features
        num_of_features = num_of_features + F.shape[1]
        feature_cell[i] = F

        # Check consistancy of features for each input images        
        assert(feature_size < 0 or feature_size == F.shape[0])
        feature_size = F.shape[0]
    
    features = np.zeros((feature_size, num_of_features))
    offset = 0
    
    # Offset and combine the features
    for i in range(num_of_imgs):
        F = feature_cell[i]
        N = F.shape[1]
        features[:,offset:offset+N] = F
        offset = offset + N
        
    return features

def extract(X, scale, filters, window, overlap, border):
    """
    :type X:        ndarray (2D)
    :type scale:    int
    :type filters:  dict
    :type window:   List[int]
    :type overlap:  List[int]
    :type border:   List[int]  
    :rtype features ndarray (2D)  
    """
    # Compute the grid for all filters
    grid = sampling_grid(X.shape, window, overlap, border, scale)
    feature_size = window[0] * window[1] * len(filters) * scale * scale

    # Current image features extraction [featrue x index]
    if not filters:
        f = indexOut(X, grid)
        features = f.reshape((f.shape[0]*f.shape[1], f.shape[2]))
    else:
        features = np.zeros((feature_size,grid.shape[2]))
        for i in range(len(filters)):
            # Filter the input image
            f = conv2d(X, filters[i+1], mode = 'same')
            # Extract the features out of the filtered image
            f = indexOut(f, grid)            
            f = f.reshape((f.shape[0]*f.shape[1], f.shape[2]))
            features[(i*f.shape[0]):((i+1)*f.shape[0]), : ] = f
    return features

def sampling_grid(img_size, window, overlap, border, scale):
    """
    :type img_size  tuple[int]
    :type window:   List[int]
    :type overlap:  List[int]
    :type border:   List[int] 
    :type scale:    int
    :rtype grid:    ndarray (3D)   
    """

    # Resize for the scale
    window = [ i * scale for i in window]
    overlap = [ i * scale for i in overlap]
    border = [ i * scale for i in border]
    
    # Create sampling grid for overlapping window
    idx = np.array(range(img_size[0]*img_size[1])).reshape(img_size)
    grid = idx[0:window[0], 0:window[1]].reshape((window[0],window[1],1))

    # Compute offsets for grid's displacement
    stride = [window[0]-overlap[0], window[1]-overlap[1]]
    offset = idx[border[0]:img_size[0]-window[0]-border[0]:stride[0], border[1]:img_size[1]-window[1]-border[1]:stride[1]]
    offset = offset.reshape((1,1,offset.size))
    
    # Prepare 3D grid - should be used as: sampled_img = img(grid)
    grid = np.tile(offset, [window[0], window[1], 1]) + np.tile(grid,[1, 1, offset.size])
    return grid
    
def indexOut(img, grid):
    """
    :type img:  ndarray (2D)
    :type grid:    ndarray (3D)
    :rtype reImg: ndarray (3D)
    """
    # To achieve Matlab-like indexing behavior
    M = img.shape[0]
    reImg = np.zeros(grid.shape)
    for layer in range(grid.shape[0]):
        for row in range(grid.shape[1]):
            for col in range(grid.shape[2]):
                idx = grid[layer][row][col]
                reImg[layer][row][col] = img[idx%M][idx//M]
    return reImg
    