#first make


CC = g++
NVCC = /usr/local/cuda-8.0/bin/nvcc
CU_ARCH= -gencode arch=compute_61,code=sm_61
EXEC = common


CU_EXEC = cudaf
TIME_EXEC = timef
TOOLS_EXEC = toolsf

all: cudaf timef toolsf

cudaf:
	$(NVCC) -std=c++11 -c cuda/*.cu
	$(NVCC) -std=c++11 -gencode arch=compute_35,code=sm_35 cuhelper.o -o $(CU_EXEC)
	
timef:
	$(CC) -std=c++11 time/*.cpp -o $(TIME_EXEC)

toolsf:
	$(CC) -std=c++11 tools/*.cpp -o $(TOOLS_EXEC)
	
	
clean:
	rm -rf $(TIME_EXEC)
	rm -rf $(TOOLS_EXEC)
	rm -rf $(CU_EXEC)
	rm -rf *.o


