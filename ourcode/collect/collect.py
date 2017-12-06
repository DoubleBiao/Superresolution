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
    
    num_of_imgs = len(images)
    feature_cell = {}
#    feature_cell = np.zeros((num_of_imgs, 1))
    num_of_features = 0
    
#    feature_size = []
    feature_size = -1
#    h = []
    
#    print("Num of imgs: {}\n".format(num_of_imgs))
    for i in range(num_of_imgs):
        # h = progress(h, i / num_of_imgs, verbose), useless, purely for documentation
        
        # sz is used for printing out info.
        # sz = images[i].shape
        
        print("For image {} \n".format(i))
        print("The size of the image is {}\n".format(images[i].shape))
        
        F = extract(images[i], scale, filters, window, overlap, border)
        num_of_features = num_of_features + F.shape[1]
        feature_cell[i] = F
        
        print("feature size: {}, F shape: {}\n".format(feature_size,F.shape))
        
#        assert(len(feature_size) == 0 or feature_size == F.shape[0])
        assert(feature_size < 0 or feature_size == F.shape[0])
        feature_size = F.shape[0]
        
    print("Feature_size: {}\n, Num of features: {}\n".format(feature_size,num_of_features))
    print("F's size is {}\n".format(F.shape))
    
    features = np.zeros((feature_size, num_of_features))
    offset = 0
    
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
    # compute one grid for all filters
    grid = sampling_grid(X.shape, window, overlap, border, scale)
    feature_size = window[0] * window[1] * len(filters) * scale * scale

    # Current image features extraction [featrue x index]
    if not filters:
        # f = X[grid]
        f = indexOut(X, grid)
        features = f.reshape((f.shape[0]*f.shape[1], f.shape[2]))
    else:
        features = np.zeros((feature_size,grid.shape[2]))
        for i in range(len(filters)):
            print(i)
#            print(filters[i+1])
#            print(grid)
            f = conv2d(X, filters[i+1], mode = 'same')
            
            print("Shape of f: {}\nShape of grid: {}".format(f.shape, grid.shape))
            
            # f = f[grid]
            f = indexOut(f, grid)
            
            print("New shape of f:{}\n".format(f.shape))
            
            f = f.reshape((f.shape[0]*f.shape[1], f.shape[2]))
            
            print("Shape of f after reshaping: {}\n".format(f.shape))
            print("Shape of feature:{} \n".format(features.shape))
            
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
    window = [ i * scale for i in window]
    overlap = [ i * scale for i in overlap]
    border = [ i * scale for i in border]
#    window[0] *= scale
#    window[1] *= scale
#    overlap[0] *= scale 
#    overlap[1] *= scale
#    border[0] *= scale
#    border[1] *= scale
    
    print("Window size is: {}\n".format(window))
    # Create sampling grid for overlapping window
    idx = np.array(range(img_size[0]*img_size[1])).reshape(img_size)
    
    # don't know why there is an element-wise deduction in the source code
    grid = idx[0:window[0], 0:window[1]].reshape((window[0],window[1],1))
#    print("idx:{} \ngrid:{}".format(idx, grid))

    # Compute offsets for grid's displacement
    # skip in the source code
    stride = [window[0]-overlap[0], window[1]-overlap[1]]
    offset = idx[1+border[0]:img_size[0]-window[0]+1-border[0]:stride[0], 1+border[1]:img_size[1]-window[1]+1-border[1]:stride[1]]
#    offset = idx[border[0]:img_size[0]-window[0]-border[0]:stride[0], border[1]:img_size[1]-window[1]-border[1]:stride[1]]
    offset = offset.reshape((1,1,offset.size))
    
    print("Offset size: {}\n".format(offset.size))
    # Prepare 3D grid - should be used as: sampled_img = img(grid)
    # grid = np.matlib.repmat(offset, window[0], window[1]) + np.matlib.repmat(grid, 1, offset.size)
    print("Shape 1: {}\nShape 2: {}\n".format(np.tile(offset, [window[0], window[1], 1]).shape,np.tile(grid,[1, 1, offset.size]).shape))
    print("Offset shape: {}, grid shape: {}\n".format(offset.shape, grid.shape))
    
    grid = np.tile(offset, [window[0], window[1], 1]) + np.tile(grid,[1, 1, offset.size])#.reshape(9,9,3180)
    return grid
    
def indexOut(img, grid):
    """
    :type img:  ndarray (2D)
    :type grid:    ndarray (3D)
    :rtype reImg: ndarray (3D)
    """
    M = img.shape[0]
    reImg = np.zeros(grid.shape)
    for layer in range(grid.shape[0]):
        for row in range(grid.shape[1]):
            for col in range(grid.shape[2]):
                idx = grid[layer][row][col]
                reImg[layer][row][col] = img[idx%M][idx//M]
    return reImg
    