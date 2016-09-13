#include "common/CudaHelper.h"
#include "common/ArgParser.h"
#include "mlas_core/mlas_matop.h"
#include "mlas_core/mlas_config.h"

void mm_test(){
	cutil::setActiveDevice(0);
	float *dA,*dB, *dC;
	unsigned int m = 1024;
	unsigned int n = 1024;
	unsigned int k = 1024;

	cutil::allocDevMem<float,unsigned int>(&dA,sizeof(float) * m*n, "Error allocating device memory for dA");
	cutil::allocDevMem<float,unsigned int>(&dB,sizeof(float) * n*k, "Error allocating device memory for dB");
	cutil::allocDevMem<float,unsigned int>(&dC,sizeof(float) * m*k, "Error allocating device memory for dC");

	cutil::cudaRandInit<float,unsigned int>(dA,m*n);
	cutil::cudaRandInit<float,unsigned int>(dB,n*k);

	dim3 mgrid((m-1)/TILE + 1, (k-1)/4 + 1, 1);
	dim3 mblock(TILE,4,1);


	Time<millis> t;
	t.start();
	mlas::sgemm<float,unsigned int,TILE><<<mgrid,mblock>>>(dA,dB,dC,m,n,k);
	cutil::handleDeviceErrors(cudaDeviceSynchronize(),"Error executing sgemm");
	double tt= t.lap();
	double flops = 2*m*n*k;
	double gflops = ((flops)/(tt/1000))/1000000000;
	std::cout << "GFLOPS:" << gflops << std::endl;



	cudaFree(dA); cudaFree(dB); cudaFree(dC);
	cudaDeviceReset();
}

int main(int argc,char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);

	if(ap.count()== 0){
		ap.menu();
		return 0;
	}

	//va_test();
	mm_test();

	return 0;
}
