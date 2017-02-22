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
#include <stdlib.h>
#include <string.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 128 
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  int index;
  double temp1;
  double temp2; 

  int jLda;
  int kLda;
  __m256d ymm0,ymm1,ymm2,ymm3, ymm8,ymm9,ymm10,ymm11, ymmTotal0;
 // int  col_reduced = K-K%16;
 // int  col_reduced32 = K-K%8;
  double scratchpad[4];

 /* int size = M*K;
  double T[size];
  for(int i=0;i<M;i++){
	for(int k =0;k<K;k++){
		index = k+i*K;
    		T[index] = A[i+k*lda];
   	}
  }
*/
  /* For each row i of A */
  for (int j = 0; j < N ; ++j)
  {  /* For each column j of B */
    for (int i  = 0; i < M; ++i) 
    {
      /* Compute C(i,j) */
      // double cij = C[i+j*n];
      int inx = i+j*lda;
      double cij = C[inx];
      temp1 = 0;
      ymmTotal0 = _mm256_broadcast_sd(&temp1);
      for( int k = 0; k < K-K%16; k+=16 )
      {  
   	//cij += T[i*K+k] * B[j*lda+k];	
        // cij += A[i+ M*k]+B[j*N+k];
        ymm8  = _mm256_loadu_pd(&A[i*lda+k]);
	ymm9  = _mm256_loadu_pd(&A[i*lda+k+4]);
	ymm10 = _mm256_loadu_pd(&A[i*lda+k+8]);
	ymm11 = _mm256_loadu_pd(&A[i*lda+k+12]);
	
	ymm0  = _mm256_loadu_pd(&B[j*lda+k]);
        ymm1  = _mm256_loadu_pd(&B[j*lda+k+4]);
        ymm2  = _mm256_loadu_pd(&B[j*lda+k+8]);
        ymm3  = _mm256_loadu_pd(&B[j*lda+k+12]);
  	
	ymm0 = _mm256_mul_pd(ymm0, ymm8 );
        ymm1 = _mm256_mul_pd(ymm1, ymm9 );
        ymm2 = _mm256_mul_pd(ymm2, ymm10);
        ymm3 = _mm256_mul_pd(ymm3, ymm11);

        ymm0 = _mm256_add_pd(ymm0, ymm1);
        ymm2 = _mm256_add_pd(ymm2, ymm3);
       	 
	ymm0 = _mm256_add_pd(ymm0,ymm2);
	ymmTotal0 = _mm256_add_pd(ymmTotal0, ymm0);
//	_mm256_storeu_pd(scratchpad, ymm0);
  //      for (int x=0; x<4; x++)
   //     { 
//	  cij += scratchpad[x];
//	}
      }
      for( int k = K-K%16; k <K-K%8; k+=8)
      {  
   	//cij += T[i*K+k] * B[j*lda+k];	
        // cij += A[i+ M*k]+B[j*N+k];
        ymm8 = _mm256_loadu_pd(&A[i*lda+k]);
	ymm9 = _mm256_loadu_pd(&A[i*lda+k+4]);
	
	ymm0  = _mm256_loadu_pd(&B[j*lda+k]);
        ymm1  = _mm256_loadu_pd(&B[j*lda+k+4]);
  	
	ymm0 = _mm256_mul_pd(ymm0, ymm8 );
        ymm1 = _mm256_mul_pd(ymm1, ymm9 );
	
	ymm0 = _mm256_add_pd(ymm0,ymm1);
        ymmTotal0 = _mm256_add_pd(ymmTotal0, ymm0);

//	_mm256_storeu_pd(scratchpad, ymm0);
  //      for (int x=0; x<4; x++)
   //    { 
//	  cij += scratchpad[x];
//	}
      }
       for( int k = K-K%8; k < K-K%4; k+=4)
      {  
   	//cij += T[i*K+k] * B[j*lda+k];	
        // cij += A[i+ M*k]+B[j*N+k];
        ymm8 = _mm256_loadu_pd(&A[i*lda+k]);
//	ymm9 = _mm256_loadu_pd(&A[i*lda+k+4]);
	
	ymm0  = _mm256_loadu_pd(&B[j*lda+k]);
  //      ymm1  = _mm256_loadu_pd(&B[j*lda+k+4]);
  	
	ymm0 = _mm256_mul_pd(ymm0, ymm8 );
    //    ymm1 = _mm256_mul_pd(ymm1, ymm9 );
	
//	ymm0 = _mm256_add_pd(ymm0,ymm1);
        ymmTotal0 = _mm256_add_pd(ymmTotal0, ymm0);
      }
     _mm256_storeu_pd(scratchpad, ymmTotal0);
     for (int x=0; x<4; x++)
     { 
	  cij += scratchpad[x];
     }
     for(int k=K-K%4;k<K;k++){
	 cij +=A[i*lda+k]*B[j*lda+k];	
     }
     C[inx] = cij;
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
  int size = lda*lda;
  double T[size];
  for(int i=0;i<lda;i++){
	for(int k =0;k<lda;k++){
		T[k+i*lda] = A[i+k*lda];
   	}
  }
  for (int j = 0; j < lda; j += BLOCK_SIZE)
  {  	
	/* For each block-column of B */
     	/* Correct block dimensions if block "goes off edge of" the matrix */
        int jLda = j*lda;
      	int N = min (BLOCK_SIZE, lda-j);
      	for (int i = 0; i < lda; i += BLOCK_SIZE)
      	{	
		/* Accumulate block dgemms into block of C */
	     	/* Correct block dimensions if block "goes off edge of" the matrix */
		/* Correct block dimensions if block "goes off edge of" the matrix */
		int M = min (BLOCK_SIZE, lda-i);			
		double* temp3 = C+i+jLda;
		for (int k = 0; k < lda; k += BLOCK_SIZE)
	      	{
			int K = min (BLOCK_SIZE, lda-k);                
			double* temp2 = B+k+jLda;
			/* Perform individual block dgemm */
			double* temp1 = T+k+i*lda;
			do_block(lda, M, N, K, temp1, temp2, temp3);
		}
	}	
  }	
}
