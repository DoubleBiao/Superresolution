# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include "batchCUBLAS.h"
# define IDX2C(i,j,ld) ((( j )*( ld ))+( i ))

float **gpuA_arr = NULL;
float **gpuA0;
float **gpuB_arr = NULL;
float **gpuB0;
float **gpuC_arr = NULL;
float **gpuC0;
int batchsize;
int atomsize;
int atomnum;
int sparsenum;
int blocksize;    

int *gammaindtab_arr;
float *b_arr;

float *dict;
float *sigs;
float *prkdx;
float *gamma_arr;


#define BATCHSIZE 2048

__global__ void _findmax(const float * inputmatrix, 
                            const int matrow, const int matcol,  
                            int * outputtable,
                            const int _elnum, const int _sparsenum)
{
    
    
    int row = blockIdx.x;

    int ty = threadIdx.x;

    int remainlen = matcol;
    int searchlen;
    int oddflag = 0;

    __shared__ float row_vec[BATCHSIZE];
    __shared__ int max_index[BATCHSIZE];
    
    __syncthreads();
    
    float maxval = 1e-20;
    int maxindx = 10086;
    int startind = 0;
    int strid;

   
    while(remainlen > 0)
    {
    
        __syncthreads();
          
        //compute the start pt from which to fetch data
        //compute the stride
             //if not at the end stride = BATCHSIZE/2
             //else stride = rest data len /2 
             //if rest data len is odd set the odd flag
  
        if(remainlen >= BATCHSIZE)
            searchlen = BATCHSIZE;
        else
        {
            searchlen = remainlen;
	    if((int)fmod((float)searchlen,(float)2) == 1)
                oddflag = 1;
        }         
        remainlen -= searchlen;

        strid = searchlen / 2;
        //load data
        if( ty < strid)
        {
            row_vec[2 * ty] = fabsf(inputmatrix[row * matcol + 2*ty + startind]);
            max_index[2 * ty] = 2*ty + startind;
            row_vec[2*ty+1] = fabsf(inputmatrix[row * matcol + 2*ty + 1 + startind]);
            max_index[2*ty+1] = 2*ty + 1 + startind;
        }
        
        //if odd flag == 1
        //add a extreme val at the end of the data array
        if(oddflag == 1 &&ty == 0)
        {     
            row_vec[searchlen - 1] = fabsf(inputmatrix[row * matcol + (searchlen - 1) + startind]);
            max_index[searchlen - 1] = searchlen - 1 + startind;
            row_vec[searchlen] = 1e-10;
            max_index[searchlen] = 10086; 
        }
        if(oddflag == 1)
        {     
            strid += 1; 
        }

        //do comparison    
        while(strid >= 1)
        {
            __syncthreads();
            if(ty < strid)
            {
                if(row_vec[ty] > row_vec[ty+strid])
                {
                    row_vec[ty] = row_vec[ty];
                    max_index[ty] = max_index[ty];
                }
                else
                {
                    row_vec[ty] = row_vec[ty+strid];
                    max_index[ty] = max_index[ty+strid];
                }
            }
                
            if(strid == 1)
                strid = 0;
            else
                strid = (strid - 1)/2 + 1 ;
        }
                
        __syncthreads();
        if(ty == 0)
        {

            //compare with the max val
            //renew the max val and max ind
 
            if(row_vec[0] > maxval )
            {
                maxval = row_vec[0];
                maxindx = max_index[0];
                
            }
        }

        startind += searchlen;

    }
    
    if(ty == 0)
    {
        outputtable[_elnum + row * _sparsenum] = maxindx;    
    }
    
}


//copy atoms in the dictionray to compute array
//each thread process one batch
__global__ void  _buildc(
float  **dense,   //the pointer to the dictionary
float  *sparse,   //the pointer to each batch matrix
int * _gammaindtab, //table indicates the atom indice to be copied
const int _sparsenum, 
const int _wrkonsize,
const int _atomnum  //number of atoms to be copied for each batch
)
{

    //determine the batch and stop some overdue thread 
    int batchind = blockDim.x * blockIdx.x + threadIdx.x;
    float *srcpt = NULL;
    float *dstpt = NULL;
    int atomind;
    
    if(batchind < _wrkonsize)
    {
        //read the wrkontab and choose the batch matrix pointer
        srcpt = dense[batchind];
        for(int i = 0; i<_atomnum; i++)
        {
            *(sparse + batchind * _atomnum + i) = 0;
        }        
        //enter in the loop
        for(int i = 0; i < _sparsenum; i++)
        {
            
            //choose the atom pointer
            atomind = _gammaindtab[batchind * _sparsenum + i];  
            dstpt = sparse + atomind + batchind * _atomnum; //dst[batchind]; 
            //read 
            *dstpt = srcpt[i];
            __syncthreads();   
        }
    }
}

//copy atoms in the dictionray to compute array
//each thread process one batch
__global__ void 
_memcpy_atom(
float  *src,   //the pointer to the dictionary
float **dst,   //the pointer to each batch matrix
int * _gammaindtab, //table indicates the atom indice to be copied
const int _sparsenum, 
const int _wrkonsize,
const int _atomnum,  //number of atoms to be copied for each batch
const int _atomsize //atom size
)
{

    //determine the batch and stop some overdue thread 
    int batchind = blockDim.x * blockIdx.x + threadIdx.x;
    float *srcpt = NULL;
    float *dstpt = NULL;
    int atomind;
    
    if(batchind < _wrkonsize)
    {
        //read the wrkontab and choose the batch matrix pointer
        dstpt = dst[batchind]; 
        
        //enter in the loop
        for(int i = 0; i < _atomnum; i++)
        {
            
            //choose the atom pointer
            atomind = _gammaindtab[batchind * _sparsenum + i];  
            srcpt = src + atomind *_atomsize;
            //read 
            for(int j = 0; j < _atomsize; j++)
            {
                dstpt[j] = srcpt[j];
            }
            __syncthreads();   
            dstpt += _atomsize; 
        }
    }
}

__global__ void 
_memcpy_sig(
float **src,   //the pointer to the dictionary
float *dst,   //the pointer to each batch matrix
const int _wrkonsize,
const int _atomsize //atom size
)
{

    //determine the batch and stop some overdue thread 
    int batchind = blockDim.x * blockIdx.x + threadIdx.x;
    float *srcpt = NULL;
    float *dstpt = NULL;
    
    if(batchind < _wrkonsize)
    {
        //read the wrkontab and choose the batch matrix pointer
        dstpt = dst + batchind*_atomsize;//dst[batchind]; 
        
            
        srcpt = src[batchind];
        //read 
        for(int j = 0; j < _atomsize; j++)
        {
            dstpt[j] = srcpt[j];
        }
        __syncthreads();   
    }
}


//each thread process one batch
__global__ void 
_memcpy_b(
float  *src,   //the pointer to the dictionary
float **dst,   //the pointer to each batch matrix
const int _wrkonsize,
const int _atomsize //atom size
)
{

    //determine the batch and stop some overdue thread 
    int batchind = blockDim.x * blockIdx.x + threadIdx.x;
    float *srcpt = NULL;
    float *dstpt = NULL;
    
    if(batchind < _wrkonsize)
    {
        //read the wrkontab and choose the batch matrix pointer
        dstpt = dst[batchind]; 
        
            
        srcpt = src + batchind *_atomsize;
        //read 
        for(int j = 0; j < _atomsize; j++)
        {
            dstpt[j] = srcpt[j];
        }
        __syncthreads();   
    }
}


//function calls _memcpy_atom
void memcpy_atom(int _blocksize,int elnum)
{
    //compute grid size
    int blockperGrid = (batchsize - 1)/_blocksize + 1;
    
    //do loading 
    
    _memcpy_atom<<<blockperGrid,_blocksize>>>
    (
    dict,   //the pointer to the dictionary
    gpuA_arr,   //the pointer to each batch matrix
    gammaindtab_arr, //table indicates the atom indice to be copied
    sparsenum,
    batchsize,
    elnum,  //number of atoms to be copied for each batch
    atomsize //atom size
    );

}

void CLEANUP() 
{       

                               
        for(int i = 0; i < batchsize; ++i)
        {    
            if(gpuA0[i]) cudaFree(gpuA0[i]);    
            if(gpuB0[i]) cudaFree(gpuB0[i]);    
            if(gpuC0[i]) cudaFree(gpuC0[i]);   
        }

        if (gpuA0) free(gpuA0);                 
        if (gpuB0) free(gpuB0);                 
        if (gpuC0) free(gpuC0);                 
        if (gpuA_arr) cudaFree(gpuA_arr);       
        if (gpuB_arr) cudaFree(gpuB_arr);       
        if (gpuC_arr) cudaFree(gpuC_arr);
        
        cudaFree(dict);
        cudaFree(sigs);
        cudaFree(prkdx);
        cudaFree(gammaindtab_arr); 
        cudaFree(b_arr);
        cudaFree(gamma_arr);
}

cublasHandle_t handle ; // CUBLAS context

int __init(int _atomsize,int _atomnum, int _batchsize, int _sparsenum, int _blocksize)
{
    
    cudaError_t err1, err2, err3;
    
    atomsize = _atomsize;
    atomnum  = _atomnum;
    batchsize = _batchsize;   
    sparsenum = _sparsenum;
    blocksize = _blocksize; 
  
    gpuA0 = (float **)malloc(sizeof(*gpuA0)*batchsize);
    gpuB0 = (float **)malloc(sizeof(*gpuB0)*batchsize);
    gpuC0 = (float **)malloc(sizeof(*gpuB0)*batchsize);
    
    for(int i = 0; i < batchsize; i++)
    { 
        err1 = cudaMalloc((void **)&gpuA0[i], sizeof(gpuA0[0][0]) * atomsize * atomnum);
        err2 = cudaMalloc((void **)&gpuB0[i], sizeof(gpuB0[0][0]) * atomsize);
        err3 = cudaMalloc((void **)&gpuC0[i], sizeof(gpuC0[0][0]) * atomsize);
        if ((err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess) )
        {
           fprintf(stderr, "step1 : !!!! GPU memory allocation error\n");
           return CUBLASTEST_FAILED;
        }
    }

    err1 = cudaMalloc((void **)&gpuA_arr, sizeof(*gpuA0) * batchsize);
    err2 = cudaMalloc((void **)&gpuB_arr, sizeof(*gpuB0) * batchsize);
    err3 = cudaMalloc((void **)&gpuC_arr, sizeof(*gpuC0) * batchsize);
    if ((err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess) )
    {
        fprintf(stderr, "step2 : !!!! GPU memory allocation error\n");
        return CUBLASTEST_FAILED;
    }
    batchsize = batchsize;


    err1 = cudaMalloc((void **)&dict, sizeof(*dict)*atomsize*atomnum);
    err2 = cudaMalloc((void **)&sigs, sizeof(*sigs)*atomsize*batchsize);
    err3 = cudaMalloc((void **)&prkdx, sizeof(*prkdx)*atomnum*batchsize);
    if ((err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess))
    {
        fprintf(stderr, "step3 !!!! GPU memory allocation error\n");
        return CUBLASTEST_FAILED;
    }

    cudaMalloc((void**)&gammaindtab_arr,sizeof(*gammaindtab_arr)*batchsize*sparsenum);

    cudaMalloc((void **)&b_arr, sizeof(*b_arr)*atomsize*batchsize);
    cublasCreate (& handle ); // initialize CUBLAS context
    cudaMalloc((void **)&gamma_arr,atomnum*batchsize*sizeof(*gamma_arr) );    
    
    err1 = cudaMemcpy(gpuA_arr, gpuA0,batchsize * sizeof(*gpuA_arr), cudaMemcpyHostToDevice); 
    err2 = cudaMemcpy(gpuB_arr, gpuB0,batchsize * sizeof(*gpuB_arr), cudaMemcpyHostToDevice);
    err3 = cudaMemcpy(gpuC_arr, gpuC0,batchsize * sizeof(*gpuC_arr), cudaMemcpyHostToDevice);
    if ((err1 != cudaSuccess) ||(err2 != cudaSuccess) || (err3 != cudaSuccess))
    {
        fprintf(stderr, "step3 : !!!! GPU memory allocation error\n");
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    return EXIT_SUCCESS ;
}

int __omp(float *A, float *b, float *c,float tol,int _batchsize) 
{
    cublasStatus_t stat ; // CUBLAS functions status
    cudaError_t err1;

    int info;
    batchsize = _batchsize;

    int elnum; // the number of non-zero element
    cublasStatus_t status1, status2, status3;

    
    float alpha;
    float beta;
    
    /***************load dict and signals for compute prkdx: 
                the inner products between signal sigs an dictionary atoms*************************************/ 
    
    status1 = cublasSetMatrix(atomsize,atomnum,sizeof(*dict),A,atomsize,dict,atomsize); //a -> d_a
    status2 = cublasSetMatrix(atomsize,batchsize,sizeof(*sigs),b,atomsize,sigs,atomsize); //b -> d_b
    status3 = cublasSetMatrix(atomsize,batchsize,sizeof(*sigs),b,atomsize,b_arr,atomsize); //b -> d_b
    if ((status1 != CUBLAS_STATUS_SUCCESS)||(status2 != CUBLAS_STATUS_SUCCESS)||(status3 != CUBLAS_STATUS_SUCCESS))
    {
        fprintf(stderr, "!!!! in omp:loc1 GPU access error (write)\n");
        return CUBLASTEST_FAILED;
    }
            
 

    /************ main loop ****************************/
    for(elnum = 0; elnum < sparsenum; elnum ++)
    {
        alpha = 1; beta = 0;
        

        /***********************compute prkdx*****************************/
        stat = cublasSgemm(handle, CUBLAS_OP_T,CUBLAS_OP_N,
        atomnum,batchsize,atomsize,
        &alpha,     dict,  atomsize,   sigs, atomsize, &beta,
        prkdx, atomnum
        );
        if(stat != CUBLAS_STATUS_SUCCESS)
        {
            cudaError_t cudaStatus = cudaGetLastError();
            fprintf(stderr, "!!!! GPU program execution error:compute prkdx \n");
            return CUBLASTEST_FAILED;
        }

        /*********** find the atom corresponding to the maximum products with the sigs *********/
        /*************** and load the atom pointer to the pointer table ***********************/
        _findmax<<<batchsize,BATCHSIZE/2>>>(prkdx, 
                            4, atomnum,  
                            gammaindtab_arr,
                            elnum, sparsenum);

        /***************compute gamma*************************************/ 
        
        //load the dictionary atoms and signals vector for each batch
        // ****************** this can be improved by dedicted kernel function*****
        // memory read from gpu to gpu

        memcpy_atom(blocksize,elnum + 1);
        _memcpy_b<<<(batchsize - 1)/blocksize + 1,blocksize>>>(
        b_arr,   //the pointer to the patchsignals
        gpuB_arr,   //the pointer to each batch matrix
        batchsize,
        atomsize //atom size
        );

        
        // do the computation by cuBlas API
        stat = cublasSgelsBatched(handle,CUBLAS_OP_N,atomsize,elnum + 1,1,gpuA_arr,atomsize,gpuB_arr,atomsize, &info,NULL,batchsize);
        cudaDeviceSynchronize();

        if(stat != CUBLAS_STATUS_SUCCESS)
        {
            cudaError_t cudaStatus = cudaGetLastError();
            fprintf(stderr, "!!!! GPU program execution error : cublas Error=%d, cuda Error=%d,(%s)\n", status1, cudaStatus,cudaGetErrorString(cudaStatus));
            return CUBLASTEST_FAILED;
        }
        if(elnum == sparsenum - 1)
            break;   
        // /*************compute redisual**********************/
        
        //load data
        //load dictionary from device to device, this can be done by dedicated kernel
        //load b from device to devie

        memcpy_atom(blocksize,elnum + 1);
        _memcpy_b<<<(batchsize - 1)/blocksize + 1,blocksize>>>(
        b_arr,   //the pointer to the patch signals
        gpuC_arr,   //the pointer to each batch matrix
        batchsize,
        atomsize //atom size
        );
        
        alpha = -1;
        beta  = 1;
        
        // *************do computation***********************
        status1 = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, atomsize, 1,
                                             elnum + 1, &alpha, (const float**)gpuA_arr, atomsize,
                                             (const float**)gpuB_arr, atomnum, &beta, gpuC_arr, atomsize, batchsize);
        
        if (status1 != CUBLAS_STATUS_SUCCESS)
        {
            cudaError_t cudaStatus = cudaGetLastError();
            fprintf(stderr, "!!!! GPU program execution error : cublas Error=%d, cuda Error=%d,(%s)\n", status1, cudaStatus,cudaGetErrorString(cudaStatus));
            return CUBLASTEST_FAILED;
        } 
        
        // get output residual

        _memcpy_sig<<<(batchsize - 1)/blocksize + 1,blocksize>>>(
        gpuC_arr,   //the pointer to the residual
        sigs,   
        batchsize,
        atomsize //atom size
        );
    }
    int blockperGrid = (batchsize - 1)/blocksize + 1;
     
     _buildc<<<blockperGrid,blocksize>>>(
    gpuB_arr,   //the pointer to the dictionary
    gamma_arr,   //the pointer to each batch matrix
    gammaindtab_arr, //table indicates the atom indice to be copied
    sparsenum, 
    batchsize,
    atomnum  //number of atoms to be copied for each batch
    );
    
    err1 = cudaMemcpy(c, gamma_arr, sizeof(float)*batchsize*atomnum, cudaMemcpyDeviceToHost);
    if((err1 != cudaSuccess))
    {
        fprintf(stderr, "!!!! GPU access error in indx read(read) in A readback\n");
        return CUBLASTEST_FAILED;
    }


    printf("end of prg\n");    
    return EXIT_SUCCESS ;
}

void __release()
{
    CLEANUP();
    cublasDestroy( handle ); // destroy CUBLAS context
}

extern "C"
{
    void omp(float *A, float *B, float *C, float tol, int _batchsize)
    {
        __omp(A, B, C, tol,_batchsize);
    }

   int init(int _atomsize,int _atomnum, int _batchsize, int _sparsenum,int _blocksize)
   {
       return __init(_atomsize,_atomnum,_batchsize,_sparsenum,_blocksize);
   }
   
   void release()
   {
      __release();
   }

}

