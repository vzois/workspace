#first make

CC = g++
NVCC = /usr/local/cuda-7.5/bin/nvcc
EXEC = common

LIB_FLAGS = -lcuda -lcudart
INCLUDE_PATHS = /usr/local/cuda/include
INCUDE_LIB = /usr/local/cuda-7.5/lib64

NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35 -rdc=true
ARCH = -gencode arch=compute_35,code=sm_35

all: cc

cc: 
	$(NVCC) -std=c++11 $(ARCH) *.cu -o $(EXEC)
	
compile:
	#$(CC) -std=c++11 -c *.cpp -L $(INCLUDE_LIB) $(LIB_FLAGS) -I$(INCLUDE_PATHS)
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -c *.cu
	$(NVCC) -std=c++11 -gencode arch=compute_35,code=sm_35 *.o -o $(EXEC)
	
clean:
	rm -rf *.o
	rm -rf $(EXEC)
