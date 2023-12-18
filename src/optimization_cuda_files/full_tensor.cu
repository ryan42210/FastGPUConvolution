#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>

#include "gpu-new-forward.h"
using namespace nvcuda;

const int BLOCK_H = 16;
const int BLOCK_W = 128;
const int WARP_SIZE = 32;

__global__ void tensor_mul_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S) {
    __shared__ __half tiled_unrolled[BLOCK_H * BLOCK_W];
    __shared__ __half tiled_mask[BLOCK_H * BLOCK_W];

    const int b = blockIdx.z;
    const int tx = threadIdx.x;
    const int warp_id = threadIdx.y;

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> mask_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> unrolled_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_frag;

    wmma::fill_fragment(acc_frag, static_cast<__half>(0.0f));

    for (int stage = 0; stage < ceil((float)(C * K * K) / BLOCK_W); stage++) {
        // load mask matrix
        __syncthreads();
        for (int i = 0; i < 8; i++) {
            int block_x = (warp_id % 4) * WARP_SIZE + tx;
            int block_y = i * 2 + (warp_id / 4);
            if (blockIdx.y * BLOCK_H + block_y < M && BLOCK_W * stage + block_x < C * K * K) {
                tiled_mask[block_y * BLOCK_W + block_x] = __float2half(mask[(blockIdx.y * BLOCK_H + block_y) * (C * K * K) + BLOCK_W * stage + block_x]);
            } else {
                tiled_mask[block_y * BLOCK_W + block_x] = static_cast<__half>(0.0f);
            }
        }
        __syncthreads();
        
        for (int s = 0; s < 8; s++) {
            int block_x = (warp_id * 16) + tx % 16;
            if (blockIdx.x * BLOCK_W + block_x < H_out * W_out) {
                for (int j = 0; j < 8; j++) {
                    int block_y = (tx / 16) * 8 + j;
                    int unrolled_h = stage * 128 + s * 16 + block_y;
                    int w = blockIdx.x * 128 + block_x;
                    int out_w = w % W_out;
                    int out_h = w / W_out;
                    int c = unrolled_h / (K * K);
                    int in_c = unrolled_h % (K * K);
                    int kernel_w = in_c % K;
                    int kernel_h = in_c / K;
                    int in_w = kernel_w + out_w * S;
                    int in_h = kernel_h + out_h * S;
                    if (unrolled_h < C * K * K) {
                        tiled_unrolled[block_y * BLOCK_W + block_x] = __float2half(input[b * (C * H * W) + c * (H * W) + (in_h * W) + in_w]);
                    } else {
                        tiled_unrolled[block_y * BLOCK_W + block_x] = static_cast<__half>(0.0f);
                    }
                }
            }
            wmma::load_matrix_sync(mask_frag, tiled_mask + 16 * s, 128);
            wmma::load_matrix_sync(unrolled_frag, tiled_unrolled + 16 * warp_id, 128);
            wmma::mma_sync(acc_frag, mask_frag, unrolled_frag, acc_frag);
        }
    }

    wmma::store_matrix_sync(tiled_unrolled + 16 * warp_id, acc_frag, 128, wmma::mem_row_major);
    int block_x = warp_id * 16 + tx % 16;
    if (blockIdx.x * BLOCK_W + block_x < H_out * W_out) {
        for (int i = 0; i < 8; i++) {
            int block_y = i * 2 + tx / 16;
            if (block_y + BLOCK_H * blockIdx.y < M) {
                output[b * (M * H_out * W_out) + (block_y + BLOCK_H * blockIdx.y) * (H_out * W_out) + (blockIdx.x * BLOCK_W + block_x)] = __half2float(tiled_unrolled[BLOCK_W * block_y + block_x]);
            }
        }
    }
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int mask_width =  (ceil(C * K * K / 4.0) + 1) * 4;

    cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, M * mask_width * sizeof(float));
    float *h_mask;
    cudaMallocHost(&h_mask, sizeof(float) * M * mask_width);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < mask_width; j++) {
            if (j < mask_width) {
                h_mask[i * mask_width + j] = host_mask[i * C * K * K + j];
            } else {
                h_mask[i * mask_width + j] = 0;
            }
        }
    }
    cudaMemcpy((void *)(*device_mask_ptr), (const void *)h_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)(*device_input_ptr), (const void *)host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaFreeHost(h_mask);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    dim3 mat_grid(ceil((float)(H_out * W_out) / BLOCK_W), ceil((float)M / BLOCK_H), B);
    dim3 mat_block(32, 8, 1);

    tensor_mul_kernel<<<mat_grid, mat_block>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
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