This the submitted code of group SRBISR.
The group members include:

Xiaotong Qiu  xq2174
Yameng Xiao  yx2417
Xiaotian Hu    xh2332
------------------------------------------------------------------------------------------------------------
The code is dvided into 2 parts

The first part of code in ./cpu_version a pure python version of Dictionary learning program.
The program is consisted in three parts:
1.    learn_dict.py: the entrance of the whole program
2.    collect.py:      image partitioning and signal extracting program
3.    ksvd.py:         pure python version of ksvd program

It relies on skimage and scipy.signal packages

Please confirm these packages can be called before running them.

To run the program, cd to the ./cpu_version and type in:
python learn_dict.py

The output is the a dictionary pair which can be used to restore high-resolution image from low-resolution image.
--------------------------------------------------------------------------------------------------------

The codes in the ./gpu_version is a parallelized version of ksvd and a cpu version ksvd for comparsion. 
The list of codes:
1.    cuomp.cu:            the cuda version of batch omp
2.    batchCUBLAS.h    some declaration used in cuomp.cu
3.    cudaomp.py         batch omp class that provide the interface from cuomp.cu to python
4.    cuksvd.py             the cuda version of ksvd
5.    ksvd.py                 pure python version of ksvd
6.    test.py                  a script that calls CPU version ksvd and GPU version ksvd for comparsion

And some data used:
1.    features.npy:

	   contains the matrix of features of each image in middle resolution image set.
2.    features_pca.npy:

 contains the result matrix of features matrix after PCA.


3.    patches.npy:           This file contains the matrix of features extracted from interpolated image set.


The batch omp is implemented in C language, to compile it, type in:

nvcc cuomp.cu -o test1.so -Xcompiler -fPIC -shared -lcublas  -gencode arch=compute_35,code=sm_35 2>test.mes

the compile log can be seen in the test.mes file

After compiling the cuomp, you can run the comparison by typing :

python test.py

The time cost of CPU version ksvd and GPU version ksvd will show at the end of running.

note: the GPU version ksvd needs malloc massive memory, which may cause error and make the program shut down. When meeting the error, just run the program again. 


