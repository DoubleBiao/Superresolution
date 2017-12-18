#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:43:16 2017

@author: serafina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:25:54 2017

@author: serafina
"""

import numpy as np
import scipy as sp
from ksvd import ksvd
import cuksvd
import time 
features_r = np.load('features.npy')
print(features_r.shape)


# comment
patches_r = np.load('patches.npy')
print(patches_r.shape)


# Configuration
conf = {'scale': 3, 'level': 1, 'window': [3,3],
        'border': [1,1], 'upsample_factor': 3, 'filters': {},
        'interpolate_kernel': 'bicubic', 'overlap': [1,1]}

# Set KSVD configuration
ksvd_conf = {'iternum': 5, 'memusage': 'normal', 'dictsize': 1024,
        'Tdata': 3, 'samples': patches_r.shape[1]}


# PCA dimensionality reduction
features_pca_r = np.load('features_pca.npy')
print(features_pca_r.shape)

ksvd_conf['data'] = features_pca_r


# KSVD
# Training process 
###########################GPU VERSION#####################################
start = time.time()
conf['dict_lores'], gamma2 = cuksvd.ksvd(ksvd_conf['iternum'],ksvd_conf['dictsize'],ksvd_conf['Tdata'],ksvd_conf['data']) 

#
p_gt = np.dot(patches_r , gamma2.T)
g_gt = np.dot(gamma2 , gamma2.T)
dict_hires = np.dot(p_gt , sp.linalg.inv(g_gt))
#
conf['dict_hires'] = dict_hires; 

timecost1 = time.time() - start


###########################CPU VERSION#######################################
features_r = np.load('features.npy')
print(features_r.shape)


# comment
patches_r = np.load('patches.npy')
print(patches_r.shape)


# Configuration
conf = {'scale': 3, 'level': 1, 'window': [3,3],
        'border': [1,1], 'upsample_factor': 3, 'filters': {},
        'interpolate_kernel': 'bicubic', 'overlap': [1,1]}

# Set KSVD configuration
ksvd_conf = {'iternum': 5, 'memusage': 'normal', 'dictsize': 1024,
        'Tdata': 3, 'samples': patches_r.shape[1]}


# PCA dimensionality reduction
features_pca_r = np.load('features_pca.npy')
print(features_pca_r.shape)

ksvd_conf['data'] = features_pca_r
start = time.time()
conf['dict_lores'], gamma2 = ksvd(ksvd_conf['iternum'],ksvd_conf['dictsize'],ksvd_conf['Tdata'],ksvd_conf['data']) 

  
p_gt = np.dot(patches_r , gamma2.T)
g_gt = np.dot(gamma2 , gamma2.T)
dict_hires = np.dot(p_gt , sp.linalg.inv(g_gt))

conf['dict_hires'] = dict_hires; 

timecost2 = time.time() - start

print "GPU version cost:",timecost1
print "CPU version cost:",timecost2

print "speedratio:", timecost2/timecost1
