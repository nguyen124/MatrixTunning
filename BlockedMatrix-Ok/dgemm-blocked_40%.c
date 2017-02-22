/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32 
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
   int index;
   double temp1, temp2, cij;
   double scratchpad[4];
   int jLda,kLda;
   __m256d ymm0,ymm1,ymm2,ymm3,ymm8,ymm9,ymm10,ymm11,ymmTotal;
 
  /* For each column j of B */ 
    for (int j = 0; j < N; j++) 
    {
      	/* Compute C(i,j) */
      	//double cij = C[i+j*lda];
      	jLda = j*lda;
      	for (int i = 0; i < M-M%4; i+=4)
	{
	     cij =0;
	     ymmTotal = _mm256_broadcast_sd(&cij);		
	     for(int k = 0;k<K-K%4;k+=4){
		int temp = k+jLda;
                ymm0  = _mm256_broadcast_sd(&B[temp]);	
		ymm1  = _mm256_broadcast_sd(&B[1+temp]);	
		ymm2  = _mm256_broadcast_sd(&B[2+temp]);	
		ymm3  = _mm256_broadcast_sd(&B[3+temp]);	
		
		temp=i+k*lda;
 		ymm8   = _mm256_loadu_pd(&A[temp]);
		ymm9   = _mm256_loadu_pd(&A[temp+lda]);
		ymm10  = _mm256_loadu_pd(&A[temp+2*lda]);
		ymm11  = _mm256_loadu_pd(&A[temp+3*lda]);
		
		ymm8   = _mm256_mul_pd(ymm0,ymm8);
		ymm9   = _mm256_mul_pd(ymm1,ymm9);
		ymm10  = _mm256_mul_pd(ymm2,ymm10);
		ymm11  = _mm256_mul_pd(ymm3,ymm11);
	
		ymm8   = _mm256_add_pd(ymm8,ymm9);	
		ymm10  = _mm256_add_pd(ymm10,ymm11);	
		ymm8   = _mm256_add_pd(ymm8,ymm10);	

		ymmTotal = _mm256_add_pd(ymmTotal,ymm8);
	    }
	    
	    for(int k = K-K%4 ;k<K;k++){
        	ymm0  = _mm256_broadcast_sd(&B[k+jLda]);	
		ymm8  = _mm256_loadu_pd(&A[i+k*lda]);
		ymm8  = _mm256_mul_pd(ymm0,ymm8);
		ymmTotal = _mm256_add_pd(ymmTotal,ymm8);
	    }
	    _mm256_storeu_pd(scratchpad,ymmTotal);
	    for(int t = 0;t<4;t++)
	    {  
		C[i+t+j*lda]+=scratchpad[t];
	    }
	}


	for(int i = M-M%4;i<M;i++){
		cij = C[i+j*lda];
   		for(int k=0;k<K;k++){            
            		temp1 = B[k+jLda];
 	      		temp2 = A[i+k*lda];             
			cij += temp1*temp2;
		 }
		 C[i+j*lda] = cij;
	}

    }
 
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
 // int size = lda*lda;
 // double T[lda*lda];
 // for(int i =0;i<size;i++){
 // 	T[i]= A[(i*lda)/size+(i*lda)%size];
 // }
 
  /* For each block-row of A */ 
  
  for (int j = 0; j < lda; j += BLOCK_SIZE)
  {  	
	/* For each block-column of B */
     	/* Correct block dimensions if block "goes off edge of" the matrix */
        int jLda = j*lda;
      	int N = min (BLOCK_SIZE, lda-j);
      	for (int k = 0; k < lda; k += BLOCK_SIZE)
      	{	
		/* Accumulate block dgemms into block of C */
	     	/* Correct block dimensions if block "goes off edge of" the matrix */
                int  kLda = k*lda;
		int K = min (BLOCK_SIZE, lda-k);                
		double* temp2 = B+k+jLda;
		for (int i = 0; i < lda; i += BLOCK_SIZE)
	      	{
			/* Correct block dimensions if block "goes off edge of" the matrix */
			int M = min (BLOCK_SIZE, lda-i);			
			/* Perform individual block dgemm */
			double* temp1 = A+i+kLda;
			double* temp3 = C+i+jLda;
			do_block(lda, M, N, K, temp1, temp2, temp3);
		}
	}	
  }	
}
