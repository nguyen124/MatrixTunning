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
   __m256d ymm0,ymm1,ymm2,ymm3,ymm8,ymm9,ymm10,ymm11,
	ymmTotal0, ymmTotal1, ymmTotal2, ymmTotal3;
 
  int size = M*K;
  double T[size];
  for(int i=0;i<M-M%4;i+=4){
	for(int k =0;k<K-K%4;k+=4){
   		for(int m=0;m<4;m++){
			T[k+(i+m)*K] = A[i+(k+m)*lda];
			T[k+1+(i+m)*K] = A[i+1+(k+m)*lda];
			T[k+2+(i+m)*K] = A[i+2+(k+m)*lda];
			T[k+3+(i+m)*K] = A[i+3+(k+m)*lda];
		}	
	}

  }




   /* For each column j of B */ 
    for (int j = 0; j < N-N%4; j+=4) 
    {
      	/* Compute C(i,j) */
      	//double cij = C[i+j*lda];
      //	jLda = j*lda;
      	for (int i = 0; i < M-M%4; i+=4)
	{
	     cij =0;
	     ymmTotal0 = _mm256_broadcast_sd(&cij);		
	     ymmTotal1 = _mm256_broadcast_sd(&cij);		
	     ymmTotal2 = _mm256_broadcast_sd(&cij);		
	     ymmTotal3 = _mm256_broadcast_sd(&cij);		
	     for(int k = 0;k<K-K%4;k+=4){
		int temp;//=i+k*lda;

// 		for(int m=0;m<4;m++){
//			T[k+(i+m)*K] = A[i+(k+m)*lda];
//			T[k+1+(i+m)*K] = A[i+1+(k+m)*lda];
//			T[k+2+(i+m)*K] = A[i+2+(k+m)*lda];
//			T[k+3+(i+m)*K] = A[i+3+(k+m)*lda];
//		}	
	//	ymm8   = _mm256_loadu_pd(&A[temp]);
	//	ymm9   = _mm256_loadu_pd(&A[temp+lda]);
	//	ymm10  = _mm256_loadu_pd(&A[temp+2*lda]);
	//	ymm11  = _mm256_loadu_pd(&A[temp+3*lda]);

 		ymm8   = _mm256_loadu_pd(&T[i*K+k]);
		ymm9   = _mm256_loadu_pd(&T[(i+1)*K+k]);
		ymm10  = _mm256_loadu_pd(&T[(i+2)*K+k]);
		ymm11  = _mm256_loadu_pd(&T[(i+3)*K+k]);

		temp = k+j*lda;
                ymm0  = _mm256_broadcast_sd(&B[temp]);	
		ymm1  = _mm256_broadcast_sd(&B[1+temp]);	
		ymm2  = _mm256_broadcast_sd(&B[2+temp]);	
		ymm3  = _mm256_broadcast_sd(&B[3+temp]);	
	
		ymm0   = _mm256_mul_pd(ymm0,ymm8);
		ymm1   = _mm256_mul_pd(ymm1,ymm9);
		ymm2  = _mm256_mul_pd(ymm2,ymm10);
		ymm3  = _mm256_mul_pd(ymm3,ymm11);
	
		ymm0   = _mm256_add_pd(ymm0,ymm1);	
		ymm2  = _mm256_add_pd(ymm2,ymm3);	
		ymm0   = _mm256_add_pd(ymm0,ymm2);	
		ymmTotal0 = _mm256_add_pd(ymmTotal0,ymm0);
	
		temp = k+(j+1)*lda;
                ymm0  = _mm256_broadcast_sd(&B[temp]);	
		ymm1  = _mm256_broadcast_sd(&B[1+temp]);	
		ymm2  = _mm256_broadcast_sd(&B[2+temp]);	
		ymm3  = _mm256_broadcast_sd(&B[3+temp]);	
	
		ymm0   = _mm256_mul_pd(ymm0,ymm8);
		ymm1   = _mm256_mul_pd(ymm1,ymm9);
		ymm2  = _mm256_mul_pd(ymm2,ymm10);
		ymm3  = _mm256_mul_pd(ymm3,ymm11);
	
		ymm0   = _mm256_add_pd(ymm0,ymm1);	
		ymm2  = _mm256_add_pd(ymm2,ymm3);	
		ymm0   = _mm256_add_pd(ymm0,ymm2);	
		ymmTotal1 = _mm256_add_pd(ymmTotal1,ymm0);
  
		temp = k + (j+2)*lda;
                ymm0  = _mm256_broadcast_sd(&B[temp]);	
		ymm1  = _mm256_broadcast_sd(&B[1+temp]);	
		ymm2  = _mm256_broadcast_sd(&B[2+temp]);	
		ymm3  = _mm256_broadcast_sd(&B[3+temp]);	
	
		ymm0   = _mm256_mul_pd(ymm0,ymm8);
		ymm1   = _mm256_mul_pd(ymm1,ymm9);
		ymm2  = _mm256_mul_pd(ymm2,ymm10);
		ymm3  = _mm256_mul_pd(ymm3,ymm11);
	
		ymm0   = _mm256_add_pd(ymm0,ymm1);	
		ymm2  = _mm256_add_pd(ymm2,ymm3);	
		ymm0   = _mm256_add_pd(ymm0,ymm2);	
		ymmTotal2 = _mm256_add_pd(ymmTotal2,ymm0);

		temp = k + (j+3)*lda;
                ymm0  = _mm256_broadcast_sd(&B[temp]);	
		ymm1  = _mm256_broadcast_sd(&B[1+temp]);	
		ymm2  = _mm256_broadcast_sd(&B[2+temp]);	
		ymm3  = _mm256_broadcast_sd(&B[3+temp]);	
	
		ymm0   = _mm256_mul_pd(ymm0,ymm8);
		ymm1   = _mm256_mul_pd(ymm1,ymm9);
		ymm2  = _mm256_mul_pd(ymm2,ymm10);
		ymm3  = _mm256_mul_pd(ymm3,ymm11);
	
		ymm0   = _mm256_add_pd(ymm0,ymm1);	
		ymm2  = _mm256_add_pd(ymm2,ymm3);	
		ymm0   = _mm256_add_pd(ymm0,ymm2);	
		ymmTotal3 = _mm256_add_pd(ymmTotal3,ymm0);

	  }
	    
	    for(int k = K-K%4 ;k<K;k++){
        	ymm8  = _mm256_loadu_pd(&A[i+k*lda]);
	//	ymm8  = _mm256_loadu_pd(&T[i*K+k]);

		ymm0  = _mm256_broadcast_sd(&B[k+j*lda]);	
		ymm0  = _mm256_mul_pd(ymm0,ymm8);
		ymmTotal0 = _mm256_add_pd(ymmTotal0,ymm0);
	    
		ymm0  = _mm256_broadcast_sd(&B[k+(j+1)*lda]);	
		ymm0  = _mm256_mul_pd(ymm0,ymm8);
		ymmTotal1 = _mm256_add_pd(ymmTotal1,ymm0);
	    
		ymm0  = _mm256_broadcast_sd(&B[k+(j+2)*lda]);	
		ymm0  = _mm256_mul_pd(ymm0,ymm8);
		ymmTotal2 = _mm256_add_pd(ymmTotal2,ymm0);
	
		ymm0  = _mm256_broadcast_sd(&B[k+(j+3)*lda]);	
		ymm0  = _mm256_mul_pd(ymm0,ymm8);
		ymmTotal3 = _mm256_add_pd(ymmTotal3,ymm0);

	    }
	    _mm256_storeu_pd(scratchpad,ymmTotal0);
	    for(int t = 0;t<4;t++)
	    {  
		C[i+t+j*lda]+=scratchpad[t];
	    }
  	
	    _mm256_storeu_pd(scratchpad,ymmTotal1);
	    for(int t = 0;t<4;t++)
	    {  
		C[i+t+(j+1)*lda]+=scratchpad[t];
	    }  
	
	     _mm256_storeu_pd(scratchpad,ymmTotal2);
	    for(int t = 0;t<4;t++)
	    {  
		C[i+t+(j+2)*lda]+=scratchpad[t];
	    }  
	
	     _mm256_storeu_pd(scratchpad,ymmTotal3);
	    for(int t = 0;t<4;t++)
	    {  
		C[i+t+(j+3)*lda]+=scratchpad[t];
	    }

	}


	for(int i = M-M%4;i<M;i++){
		//cij = C[i+j*lda];
   		for(int k=0;k<K;k++){            
        		temp1 = A[i+k*lda];             		

	    		temp2 = B[k+j*lda]; 	      					
			C[i+j*lda] += temp1*temp2;
			
			temp2 = B[k+(j+1)*lda];
			C[i+(j+1)*lda] += temp1*temp2;
		
			temp2 = B[k+(j+2)*lda];
			C[i+(j+2)*lda] += temp1*temp2;
	
			temp2 = B[k+(j+3)*lda];
			C[i+(j+3)*lda] += temp1*temp2;
			//cij += temp1*temp2;
		 }
		// C[i+j*lda] = cij;
	}
    }

    for(int j = N-N%4;j<N;j++){
   	jLda = j*lda;
      	for (int i = 0; i < M-M%4; i+=4)
	{
	     cij =0;
	     ymmTotal1 = _mm256_broadcast_sd(&cij);		
	     for(int k = 0;k<K-K%4;k+=4){
		int temp = k+jLda;
                ymm0  = _mm256_broadcast_sd(&B[temp]);	
		ymm1  = _mm256_broadcast_sd(&B[1+temp]);	
		ymm2  = _mm256_broadcast_sd(&B[2+temp]);	
		ymm3  = _mm256_broadcast_sd(&B[3+temp]);	
		
		temp=i+k*lda;
 	//	ymm8   = _mm256_loadu_pd(&A[temp]);
//		ymm9   = _mm256_loadu_pd(&A[temp+lda]);
//		ymm10  = _mm256_loadu_pd(&A[temp+2*lda]);
//		ymm11  = _mm256_loadu_pd(&A[temp+3*lda]);
	
		ymm8   = _mm256_loadu_pd(&T[i*K+k]);
		ymm9   = _mm256_loadu_pd(&T[(i+1)*K+k]);
		ymm10  = _mm256_loadu_pd(&T[(i+2)*K+k]);
		ymm11  = _mm256_loadu_pd(&T[(i+3)*K+k]);
	
		ymm8   = _mm256_mul_pd(ymm0,ymm8);
		ymm9   = _mm256_mul_pd(ymm1,ymm9);
		ymm10  = _mm256_mul_pd(ymm2,ymm10);
		ymm11  = _mm256_mul_pd(ymm3,ymm11);
	
		ymm8   = _mm256_add_pd(ymm8,ymm9);	
		ymm10  = _mm256_add_pd(ymm10,ymm11);	
		ymm8   = _mm256_add_pd(ymm8,ymm10);	

		ymmTotal1 = _mm256_add_pd(ymmTotal1,ymm8);
	    }
	    
	    for(int k = K-K%4 ;k<K;k++){
        	ymm0  = _mm256_broadcast_sd(&B[k+jLda]);	
		ymm8  = _mm256_loadu_pd(&A[i+k*lda]);
		ymm8  = _mm256_mul_pd(ymm0,ymm8);
		ymmTotal1 = _mm256_add_pd(ymmTotal1,ymm8);
	    }
	    _mm256_storeu_pd(scratchpad,ymmTotal1);
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
		int K = min (BLOCK_SIZE, lda-k);                
		double* temp2 = B+k+jLda;
	        int  kLda = k*lda;
		for (int i = 0; i < lda; i += BLOCK_SIZE)
	      	{
	        	int M = min (BLOCK_SIZE, lda-i);			
			double* temp3 = C+i+jLda;
			double* temp1 = A+i+kLda;
			do_block(lda, M, N, K, temp1, temp2, temp3);
		}
	}	
  }	
}
