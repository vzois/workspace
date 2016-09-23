#include "CudaHelper.h"

#define SIZE 128

int main(){
	float *hA,*hB;
	float *dA,*dB;

	cutil::setActiveDevice(0);

	cutil::safeMalloc<float,int>(&dA,sizeof(float)*SIZE,"allocating memory for dA");
	cutil::safeMalloc<float,int>(&dB,sizeof(float)*SIZE,"allocating memory for dB");

	cutil::safeMallocHost<float,int>(&hA,sizeof(float)*SIZE,"allocating memory for hA");
	cutil::safeMallocHost<float,int>(&hB,sizeof(float)*SIZE,"allocating memory for hB");

	Utils<float> u;
	for(int i = 0;i<SIZE;i++){ hA[i] = u.uni(100); }

	cutil::safeCopyToDevice<float>(dA,hA,sizeof(float) * SIZE, "copying hA to dA");
	cutil::safeCopyToDevice<float>(dB,hB,sizeof(float) * SIZE, "copying hB to dB");

	cudaFree(dA); cudaFree(dB);
	cudaFreeHost(hA); cudaFreeHost(hB);



	return 0;
}
