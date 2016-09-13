#first make

CC = g++
NVCC = /usr/local/cuda-8.0/bin/nvcc
EXEC = mlas

#LIBRARIES#
LIB_FLAGS = -lcuda -lcudart
INCLUDE_PATHS = /usr/local/cuda/include
INCUDE_LIB = /usr/local/cuda-8.0/lib64

#COMPILATION FILES#
CPP_FILES = *.cpp
CU_FILES = *.cu

NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35
ARCH= -gencode arch=compute_52,code=sm_52

all: cc

cc:
	$(NVCC) -std=c++11 $(ARCH) *.cu -o $(EXEC)
	
compile:
	#$(CC) -std=c++11 -c $(CPP_FILES) -L $(INCLUDE_LIB) $(LIB_FLAGS) -I$(INCLUDE_PATHS)
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -c $(CU_FILES)
	$(NVCC) -std=c++11 -gencode arch=compute_35,code=sm_35 *.o -o $(EXEC)
	
	
ptx:
	$(NVCC) -std=c++11 $(ARCH) -ptx *.cu
	
clean:
	rm -rf *.o
	rm -rf $(EXEC)
