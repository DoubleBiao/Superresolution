import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import time

class cudaomp():
    def __init__(self,sopath):
        lib = ctypes.cdll.LoadLibrary(sopath)
        self.omp = lib.omp
        self.omp.argtypes = \
        [\
        ndpointer(ctypes.c_float),\
        ndpointer(ctypes.c_float),\
        ndpointer(ctypes.c_float),\
        ctypes.c_float,
        ]
        
        self.ompinit = lib.init
        self.ompinit.argtypes = \
        [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
        ]

        self.omprelease = lib.release

    def release(self):
        self.omprelease()

    def init(self,atomsize, atomnum, batchsize,sparsenum,blocksize):
        self.ompinit(atomsize, atomnum, batchsize,sparsenum,blocksize)

    def batchomp(self,A,b,c,signum,batchsize):
        A = np.asfortranarray(A)
        b = np.asfortranarray(b)
        c = np.asfortranarray(c)
        start = time.time()
        batchnum,remainsize = divmod(signum,batchsize)
        for batchind in range(batchnum):
            b_tmp =b[:,batchind*batchsize: (batchind+1)*batchsize ]
            c_tmp =c[:,batchind * batchsize : (batchind + 1)*batchsize]
            self.omp(A, b_tmp,  c_tmp   ,1e-7,batchsize)
            c[:,batchind * batchsize : (batchind + 1)*batchsize] = c_tmp
        if(remainsize != 0):
            b_tmp =b[:,batchnum*batchsize:]
            c_tmp =c[:,batchnum*batchsize:]
            self.omp(A, b_tmp,  c_tmp   ,1e-7, remainsize)
            c[:,batchnum * batchsize :] = c_tmp

        timecost = time.time() - start
        c = np.ascontiguousarray(c)
        return c,timecost


#def omp(X, y, ncoef=None, maxit=200, tol=1e-6, ztol=1e-12):
#    """ orthogonal matching pursuit. Compute the sparse representation of input 
#    y using dictionary X
#    input: 
#        X: the dictionary. Format: each column of X is a atom
#        y: the input sample. Format: a vector
#        ncoef: the maximum number of nonzero element in the sparse representation
#        maxit: maximum iteration time
#        tol: convergence tolerance
#        ztol: threshold for residual covariance
#    """
#        
#    def norm2(x):
#        return np.linalg.norm(x)/np.sqrt(len(x))
#    
#    if ncoef is None:
#        ncoef = int(X.shape[1]/2)
#    
#    X_transpose = X.T
#    Alpha = X_transpose.dot(y)
#    
#    active = []
#    coef = np.zeros(X.shape[1],dtype = float)
#    
#    residual = y
# 
#    ynorm = norm2(y)                         # store for computing relative err
#
#
#    tol = tol * ynorm       # convergence tolerance
#    ztol = ztol * ynorm     # threshold for residual covariance
#    
#     # main iteration
#    maxit = ncoef
#    for it in range(maxit):
#        
#        # compute residual covariance vector and check threshold
#        rcov = np.dot(X_transpose, residual)
#        rcov = abs(rcov)
#    #    print("prcdk:")
#    #    print(rcov)
#        i = np.argmax(rcov)
#        rc = rcov[i]
#
#        if rc < ztol:
#            #print('All residual covariances are below threshold.')
#            break
#        
#        if i not in active:
#            active.append(i)
#        
#        DI = X[:, active]
#        
#        coefi = np.dot(np.linalg.inv(DI.T.dot(DI)),Alpha[active])
#        coef[active] = coefi   # update solution
#    #    print "coefi"
#    #    print coef
#        residual = y - np.dot(X[:,active], coefi)
#  
#        err = norm2(residual) / ynorm  
#        
#        
#    #    print residual    
#            
#        # check stopping criteria
#        if err < tol:  # converged
#            break
#        if len(active) >= ncoef:   # hit max coefficients
#            break
#        if it == maxit-1:  # max iterations
#            break 
#    return coef,err
#
#
#atomsize = 30
#atomnum  = 1024
#batchsize = 2048
#signum = 4086*2
#sparsenum = 3
#blocksize = 128
#
#
#A = np.random.rand(atomsize,atomnum).astype(np.float32)
#b = np.random.rand(atomsize,signum).astype(np.float32)
#
#c = np.zeros((atomnum,signum)).astype(np.float32)
#
#
#
#cuomp = cudaomp("./test1.so")
#
#
#cuomp.init(atomsize, atomnum, batchsize,sparsenum,blocksize)
#
#
#
#c,timecost = cuomp.batchomp(A,b,c,signum,batchsize)
#print "GPU time cost:",timecost
##print " "
##print " "
#print "-----------------CPU time ------------------"
##print "A"
##print A
#print " "
##print c
#out = np.empty_like(c)
#print out.shape
#
#err = 0
#start = time.time()
#for i in range(signum):
##    print "b:"
##    print b[:,i]
##    print " "
#    out[:,i],errtmp = omp(A,b[:,i], ncoef=3)
#    err += errtmp
#print "CPU time cost:",time.time() - start
#
#print " "
#print " "
#print np.allclose(c,out,atol = 1e-3)      


    
#cuomp.release()
