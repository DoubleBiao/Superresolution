# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:28:47 2017

@author: HuXiaotian
"""

import numpy as np
from sklearn.linear_model import orthogonal_mp_gram

def omp(X, y, ncoef=None, maxit=200, tol=1e-6, ztol=1e-12):
    """ orthogonal matching pursuit. Compute the sparse representation of input 
    y using dictionary X
    input: 
        X: the dictionary. Format: each column of X is a atom
        y: the input sample. Format: a vector
        ncoef: the maximum number of nonzero element in the sparse representation
        maxit: maximum iteration time
        tol: convergence tolerance
        ztol: threshold for residual covariance
    """
        
    def norm2(x):
        return np.linalg.norm(x)/np.sqrt(len(x))
    
    if ncoef is None:
        ncoef = int(X.shape[1]/2)
    
    X_transpose = X.T
    Alpha = X_transpose.dot(y)
    
    active = []
    coef = np.zeros(X.shape[1],dtype = float)
    
    residual = y
    
 
    ynorm = norm2(y)                         # store for computing relative err


    tol = tol * ynorm       # convergence tolerance
    ztol = ztol * ynorm     # threshold for residual covariance
    
     # main iteration
    for it in range(maxit):
        
        # compute residual covariance vector and check threshold
        rcov = np.dot(X_transpose, residual)
        i = np.argmax(rcov)
        rc = rcov[i]

        if rc < ztol:
            #print('All residual covariances are below threshold.')
            break
        
        if i not in active:
            active.append(i)
        
        DI = X[:, active]
 
        coefi = np.dot(np.linalg.inv(DI.T.dot(DI)),Alpha[active])
        coef[active] = coefi   # update solution
        
        residual = y - np.dot(X[:,active], coefi)
  
        err = norm2(residual) / ynorm  
        
        
            
        # check stopping criteria
        if err < tol:  # converged
            break
        if len(active) >= ncoef:   # hit max coefficients
            break
        if it == maxit-1:  # max iterations
            break
    
    return coef



def ksvd(max_iter,n_components,transform_n_nonzero_coefs,X):
    """ ksvd function
    input:
        max_iter: the maximum iteration times
        n_components: the size of dictionary(the number of atoms)
        transform_n_nonzero_coefs: Tdata, the maximum number of nonzero elements
        in the sparse representation
        X: training data. Format: each column of the X is a training sample(patch)
    Output:
        dictionary: Format: each column of output dictionary D is a atom
        gamma: the sparse representaion of input data, Format: each column of 
        gamma is a sample
        """
        
    # replace unuse atoms

    #init dictionary
    Xnorm2 = (X**2).sum(axis = 0)
    nonzerosam = np.nonzero(Xnorm2 > 1e-3)[0]
    tol = 1e-6
    
    ###for debug
    
    #D1 = X[:,nonzerosam[0:1024]]
    D = X[:,np.random.choice(nonzerosam,n_components,replace = False)]
    
    
    D /= np.linalg.norm(D, axis=0)
    
    
    gamma = np.zeros((n_components,X.shape[1]))
    erec = np.zeros(max_iter)
    for ite in range(max_iter):
        print("The {}-th iteration is running now\n".format(ite))
        for i in range(X.shape[1]):
            gamma[:,i] = omp(D,X[:,i],ncoef=3)
 
        e = np.linalg.norm(X - D.dot(gamma),axis = 0)
        e = np.sqrt(sum(e)/e.size)
   
        erec[ ite ] = e
        print("error",e)
        if e < tol:
            break
        
        replaced_atoms = np.zeros(n_components)
        unused_sigs = np.arange(X.shape[1])
        
        p = np.arange(n_components)#np.random.permutation(n_components)
        for j in range(n_components):
            I = np.nonzero(gamma[p[j],:] > tol)[0]
            if np.sum(I) == 0:
                #continue
                perm = np.random.permutation(len(unused_sigs))
                E = np.sum((X[:,unused_sigs[perm]] - D.dot(gamma[:,unused_sigs[perm]]))**2, axis = 0)
                i = np.argmax(E)
                atom = X[:,unused_sigs[perm[i]]]
                atom = atom/np.linalg.norm(atom)
                D[:,p[j]] = atom;
                unused_sigs = np.delete(unused_sigs,perm[i])
                replaced_atoms[j] = 1;
            else:
                D[:,p[j]] = 0
                g = gamma[p[j],I].T
                Xrnm = X[:,I]
                d = np.dot(Xrnm - np.dot(D,gamma[:,I]),g)
                d = d/np.linalg.norm(d)
                g = np.dot(Xrnm.T - np.dot(D,gamma[:,I]).T,d)
                D[:,p[j]] = d
                gamma[p[j],I] = g.T
         
    return D,gamma

atomlen = 100
dictsize = 200
samplenum = 400
X = np.random.randn(atomlen,samplenum)
y = np.random.randn(atomlen)
itetimes = 5
Tdata = 6


D,gamma = ksvd(itetimes,dictsize,Tdata,X)

gt = gamma.dot(gamma.T)

print(np.linalg.matrix_rank(gt))
np.linalg.inv(gt)