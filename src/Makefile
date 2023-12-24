CC=g++
NVCC=nvcc
CXXFLAGS= -O2 -Wextra -std=c++17
CUDAFLAGS= -std=c++17 -c -arch=sm_70
LIBDIRS=-L/usr/local/cuda-7.5/lib64
INCDIRS=-I/usr/local/cuda-7.5/include

all: matrix_cuda.o
        $(CC) -o gpv_convolution main.cpp gpu_conv.cpp conv_kernel.o $(LIBDIRS) $(INCDIRS) $(CXXFLAGS)
conv_kernel.o: gpu_conv.cu
        $(NVCC) $(CUDAFLAGS)  conv_kernel.cu
clean:
    rm -rf gpv_convolution *.o