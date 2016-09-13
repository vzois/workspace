#include "../../common/CudaHelper.h"
#include "../../common/ArgParser.h"
#include "mlas_core/mlas_matop.h"
#include "mlas_core/mlas_config.h"

int multiplier = 1;

void mm_test(){
	cutil::setActiveDevice(0);
	float *dA,*dB, *dC;
	uint64_t m = 1024 * multiplier;
	uint64_t n = 1024 * multiplier;
	uint64_t k = 1024 * multiplier;

	cutil::allocDevMem<float,uint64_t>(&dA,sizeof(float) * m*n, "Error allocating device memory for dA");
	cutil::allocDevMem<float,uint64_t>(&dB,sizeof(float) * n*k, "Error allocating device memory for dB");
	cutil::allocDevMem<float,uint64_t>(&dC,sizeof(float) * m*k, "Error allocating device memory for dC");

	cutil::cudaRandInit<float,unsigned int>(dA,m*n);
	cutil::cudaRandInit<float,unsigned int>(dB,n*k);

	dim3 mgrid((m-1)/TILE + 1, (k-1)/4 + 1, 1);
	dim3 mblock(TILE,4,1);

	cutil::print_grid(mgrid,mblock);

	Time<millis> t;
	t.start();
	mlas::sgemm<float,unsigned int,TILE><<<mgrid,mblock>>>(dA,dB,dC,m,n,k);
	cutil::handleDeviceErrors(cudaDeviceSynchronize(),"Error executing sgemm");
	double tt= t.lap();
	double GBs = ((double)( m * n * 3 * 4 )) / (1000 * 1000 * 1000);
	uint64_t flop = 2 * n * m * k;
	double gflops = ((flop)/(tt/1000))/1000000000;
	std::cout << "GBs: " << GBs << std::endl;
	std::cout << "FLOP:" << flop << std::endl;
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

	if(ap.exists("-c")){
		printf("Multiplier:%d\n",ap.getInt("-c"));
		multiplier = ap.getInt("-c");
	}
	//va_test();
	mm_test();

	return 0;
}
