/*
 * CUBLAS WRAPPERS USED FOR COMPARISON TESTING
 *
 */

#ifndef MLAS_CUBLAS
#define MLAS_CUBLAS

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../common/Time.h"
#include "../../common/CudaHelper.h"

namespace mlas{

	void sgemm_test(uint64_t multiplier){
		cutil::setActiveDevice(0);
		uint64_t m,n,k;

		m = multiplier * 1024; n = multiplier * 1024; k = multiplier * 1024;


		float *dA,*dB,*dC;
		cutil::safeMalloc<float,uint64_t>(&dA,sizeof(float)*m*n,"dA memory alloc");
		cutil::safeMalloc<float,uint64_t>(&dB,sizeof(float)*n*k,"dB memory alloc");
		cutil::safeMalloc<float,uint64_t>(&dC,sizeof(float)*m*k,"dC memory alloc");

		//cutil::cudaRandInit<float,uint64_t>(dA,m*n);
		//cutil::cudaRandInit<float,uint64_t>(dB,n*k);

		cublasHandle_t handle;
		const float alpha = 1.0f;
		const float beta  = 0.0f;

		Time<secs> t;

		t.start();
		cutil::cublasCheckErr(cublasCreate(&handle), "Creating Handle Error!");
		cublasSgemm(
				handle,
				CUBLAS_OP_N,CUBLAS_OP_N,
				k,m,n,
				&alpha,
				dB,k,
				dA,n,
				&beta,
				dC,k
		);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing cublas sgemm");
		double tt=t.lap("sgemm elapsed time in ms");

		uint64_t flop = 2 * ((uint64_t)m)*((uint64_t)n)*((uint64_t)m);

		std::cout << "FLOP:" << flop << std::endl;
		double gflops = ((flop)/(tt))/1000000000;
		std::cout << "GFLOPS CUBLAS sgemm:" << gflops << std::endl;

		cutil::cublasCheckErr(cublasDestroy(handle), "Destroying Handle Error");

		cudaFree(dA);
		cudaFree(dB);
		cudaFree(dC);
		cudaDeviceReset();
	}

}


#endif


