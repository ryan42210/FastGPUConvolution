#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>

#include "gpu-new-forward.h"
using namespace nvcuda;

constexpr int TILE_HEIGHT = 8;
constexpr int TILE_WIDTH = 16;
constexpr int WARP_SIZE = 32;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void tensor_mul_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S, const int H_out, const int W_out, const int unrolled_width, const int vector_len) {
    __shared__ __half tiled_unrolled[WMMA_K * WMMA_N];
    __shared__ __half tiled_mask[WMMA_M * WMMA_K];

    const int b = blockIdx.z;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int w = blockDim.x * blockIdx.x + tx;
    const int h = TILE_WIDTH * blockIdx.y + ty;
    const int out_w = w % W_out;
    const int out_h = w / W_out;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> mask_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> unrolled_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_frag;

    wmma::fill_fragment(acc_frag, static_cast<__half>(0));

    for (int stage = 0; stage < ceil((float)vector_len / WMMA_K); stage++) {
        // load mask matrix
        if (stage * TILE_WIDTH + tx < vector_len) {
            if (h < M) {
                tiled_mask[ty * blockDim.x + tx] = __float2half(mask[h * vector_len + stage * WMMA_K + tx]);
            } else {
                tiled_mask[ty * blockDim.x + tx] = static_cast<__half>(0);
            }
            if (h + 8 < M) {
                tiled_mask[(ty + 8) * blockDim.x + tx] = __float2half(mask[(h + 8) * vector_len + stage * WMMA_K + tx]);
            } else {
                tiled_mask[(ty + 8) * blockDim.x + tx] = static_cast<__half>(0);
            }
        }

        if (w < unrolled_width) {
            int unrolled_h = (stage * TILE_WIDTH + ty);
            if (unrolled_h < vector_len) {
                int in_c = unrolled_h % (K * K);
                int in_w = (in_c % K) + out_w * S;
                int in_h = (in_c / K) + out_h * S;
                tiled_unrolled[ty * TILE_WIDTH + tx] = __float2half(input[b * (C * H * W) + (unrolled_h / (K * K)) * (H * W) + in_h * (W) + in_w]);
            } else {
                tiled_unrolled[ty * TILE_WIDTH + tx] = static_cast<__half>(0);
            }
            if (unrolled_h + 8 < vector_len) {
                int in_c = (unrolled_h + 8) % (K * K);
                int in_w = (in_c % K) + out_w * S;
                int in_h = (in_c / K) + out_h * S;
                tiled_unrolled[(ty + 8) * TILE_WIDTH + tx] = __float2half(input[b * (C * H * W) + ((unrolled_h + 8)/ (K * K)) * (H * W) + in_h * (W) + in_w]);
            } else {
                tiled_unrolled[(ty + 8) * TILE_WIDTH + tx] = static_cast<__half>(0);
            }
        }
        __syncthreads();

        if (tx + ty * TILE_WIDTH < WARP_SIZE) {
            // Load the inputs
            wmma::load_matrix_sync(mask_frag, tiled_mask, WMMA_K);
            wmma::load_matrix_sync(unrolled_frag, tiled_unrolled, WMMA_N);
            wmma::mma_sync(acc_frag, mask_frag, unrolled_frag, acc_frag);
        }
        __syncthreads();
    }

    if (tx + ty * blockDim.x < WARP_SIZE) {
        wmma::store_matrix_sync(tiled_unrolled, acc_frag, WMMA_N, wmma::mem_row_major);
    }
    __syncthreads();
    if (h < M && w < unrolled_width) {
        output[b * (M * unrolled_width) + h * (unrolled_width) + w] = __half2float(tiled_unrolled[ty * blockDim.x + tx]);
    }
    if (h + 8 < M && w < unrolled_width) {
        output[b * (M * unrolled_width) + (h + 8) * (unrolled_width) + w] = __half2float(tiled_unrolled[(ty + 8) * blockDim.x + tx]);
    }
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, M * C * K * K * sizeof(float));
    
    cudaMemcpy((void *)(*device_mask_ptr), (const void *)host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)(*device_input_ptr), (const void *)host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int out_w = H_out * W_out;

    dim3 mat_grid(ceil((float)out_w / TILE_WIDTH), ceil((float)M / TILE_WIDTH), B);
    dim3 mat_block(TILE_WIDTH, TILE_HEIGHT, 1);
    tensor_mul_kernel<<<mat_grid, mat_block>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S, H_out, W_out, out_w, C*K*K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}