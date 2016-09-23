#ifndef MLAS_VECOP_H
#define MLAS_VECOP_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mlas{
	template<typename DATA_T, typename SIZE_T, unsigned int TILE>
	__global__ void sgemm(
			DATA_T *A,
			DATA_T *B,
			DATA_T *C,
			SIZE_T m,
			SIZE_T n,
			SIZE_T k
		){
			__shared__ DATA_T sA[TILE * TILE];
			__shared__ DATA_T sB[TILE * TILE];

			int row = ( blockIdx.y * blockDim.y + threadIdx.y );
			int col = ( blockIdx.x * blockDim.x + threadIdx.x );
			DATA_T rC = 0;

			for(int i = 0; i < (n + 1)/TILE; i++){
					sA[threadIdx.y*TILE + threadIdx.x] = A[ row * (n+1) + i * TILE + threadIdx.x ];
					sB[threadIdx.y*TILE + threadIdx.x] = B[ (i * TILE + threadIdx.y) * k  + col];
					__syncthreads();
					for(int j = 0;j< TILE; j++){
						//rC += sA[threadIdx.y * TILE + j] * sB[j * TILE + threadIdx.x]; // 354 GFLOPS
						rC += sA[threadIdx.y * TILE + j] * sB[threadIdx.y * TILE + j]; // 454 GFLOPS
					}
					__syncthreads();
			}

			C[row * k + col] = rC;
	}

	template<typename DATA_T, typename SIZE_T, unsigned int TILE_X, unsigned int TILE_Y>
	__global__ void sgemm_32x8x4(
			DATA_T *A, DATA_T *B, DATA_T *C,
			SIZE_T m, SIZE_T n, SIZE_T k
		){
			__shared__ DATA_T sA[1024];
			__shared__ DATA_T sB[1024];

			int row = ( blockIdx.y * blockDim.y + threadIdx.y );
			int col = ( blockIdx.x * blockDim.x + threadIdx.x );

			if( TILE_X == 32 && TILE_Y == 32){ // 1024 threads, 1 element per thread
				DATA_T rC = 0;
				sA[threadIdx.y*TILE_X + threadIdx.x] = A[ row * (n+1) + TILE_X + threadIdx.x ];
				sB[threadIdx.y*TILE_X + threadIdx.x] = B[ ( TILE_X + threadIdx.y ) * k  + col];
				__syncthreads();

			}else if(TILE_X == 32 && TILE_Y == 8){ // 256 threads, 8 elements per thread


			}



	}
}


#endif
