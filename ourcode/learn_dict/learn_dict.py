#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:34:27 2017

@author: serafina
"""

import numpy as np
import scipy as sp
from skimage import color,io,img_as_float,transform

def convert_ycbcr_y(f):
    rgb = io.imread(f)
    ycbcr = color.rgb2ycbcr(rgb)
    ycbcr = img_as_float(ycbcr)/255
    return ycbcr[:,:,0]
    

str = 'CVPR08-SR/Data/Training' + '/*.bmp'

load_imgs = io.ImageCollection(str,load_func = convert_ycbcr_y)

#io.imshow(coll_img[40])

# Configuration
conf = {'scale': 3, 'level': 1, 'window': [3,3],
        'border': [1,1], 'upsample_factor': 3, 'filters': {},
        'interpolate_kernel': 'bicubic', 'overlap': [2,2]}

O = np.zeros((conf['upsample_factor']-1))
#G = np.array([1,O,-1]); # Gradient
G = np.array([1])
G = np.concatenate((G,O))
G = np.append(G,-1)
G.shape = (1,2+O.size)
#L = np.array([1,O,-2,O,1])/2; # Laplacian
L = np.array([1])
L = np.concatenate((L,O))
L = np.append(L,-2)
L = np.concatenate((L,O))
L = np.append(L,1)
L = L/2.0;
L.shape = (1,3+2*O.size)

filters = {'filter1':G,'filter2':G.T,'filter3':L,'filter4':L.T}
conf['filters'] = filters

def modcrop(imgs,scale):
    imgs_c = []
    for i in range(len(imgs)):
        img = imgs[i]
        sz = img.shape
        sz = sz - np.mod(sz,scale)
        img_c = img[0:sz[0], 0:sz[1]]
        imgs_c.append(img_c)
    return imgs_c
    
hires = modcrop(load_imgs, conf['scale'])
 
  
lores = []
for i in range(len(hires)):
    img = hires[i]
    sz = img.shape
    scale = conf['scale']
    img_l = transform.resize(img,(sz[0]/scale,sz[1]/scale))    
    lores.append(img_l)
#lores = np.asarray(lores)
    
midres = []
for i in range(len(lores)):
    img = lores[i]
    sz = img.shape
    scale = conf['upsample_factor']
    img_m = transform.resize(img,(sz[0]*scale,sz[1]*scale))    
    midres.append(img_m)
#midres = np.asarray(midres)
    
# COLLECT 
features = collect(midres, conf['scale'], conf['filters'], conf['window'], conf['overlap'], conf['border']);
   
interpolated = []
for i in range(len(lores)):
    img = lores[i]
    sz = img.shape
    scale = conf['scale']
    img_i = transform.resize(img,(sz[0]*scale,sz[1]*scale))    
    interpolated.append(img_i)

patches = []
for i in range(len(hires)):
    img_h = hires[i]
    img_l = interpolated[i]
    img_p = img_h - img_l
    patches.append(img_p)
#patches = np.asarray(patches)    
    
# COLLECT    
patches = collect(patches, conf['scale'], {}, conf['window'], conf['overlap'], conf['border']);

# comment    
#features_1 = sp.io.matlab.mio.loadmat('features.mat')
#features = features_1['features']

# comment
#patches_1 = sp.io.matlab.mio.loadmat('patches.mat') 
#patches = patches_1['patches']
    
# Set KSVD configuration
ksvd_conf = {'iternum': 20, 'memusage': 'normal', 'dictsize': 1024,
        'Tdata': 3, 'samples': patches.shape[1]}



# PCA dimensionality reduction
C = np.dot(features , features.T)
V,D = np.linalg.eig(C); # V:eigenvalue; D:eigen vector
sorted_indices = np.argsort(V)
evals = V[sorted_indices]
evecs = D[:,sorted_indices]

evals = np.cumsum(evals) / np.sum(evals)
b = np.arange(len(evals))
c = b[evals>1e-3]
k = c[0]
end = evals.shape[0]
evals.shape = (end,1)
conf['V_pca'] = evecs[:, k:end]; 
conf['ksvd_conf'] = ksvd_conf;
features_pca = np.dot(conf['V_pca'].T , features)
    
ksvd_conf['data'] = features_pca

# KSVD
# Training process 
conf['dict_lores'], gamma = ksvd(ksvd_conf['iternum'],ksvd_conf['dictsize'],ksvd_conf['Tdata'],ksvd_conf['data']) 
  
# comment    
#gamma_1 = sp.io.matlab.mio.loadmat('gamma.mat')
#gamma = gamma_1['gamma']
#
p_gt = np.dot(patches , gamma.T)
g_gt = np.dot(gamma , gamma.T)
dict_hires = np.dot(p_gt , sp.linalg.inv(g_gt))

conf['dict_hires'] = dict_hires;     
    
    
    
    
    
    
    
    
    
    
    
    
    
