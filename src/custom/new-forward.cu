#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>

#include "gpu-new-forward.h"
using namespace nvcuda;

constexpr int BLOCK_H = 16;
constexpr int BLOCK_W = 128;

__global__ void tensor_mul_kernel(float * __restrict__ output, const float * __restrict__ input, const __half * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S) {
    __shared__ __half tiled_unrolled[BLOCK_H * BLOCK_W];

    const int b = blockIdx.z;
    const int tx = threadIdx.x;
    const int warp_id = threadIdx.y;

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int mask_width =  ceilf((float)C * K * K / 8.0f) * 8;


    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> mask_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> unrolled_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_frag;
    wmma::fill_fragment(acc_frag, static_cast<__half>(0.0f));

    for (int stage = 0; stage < ceilf((float)(C * K * K) / BLOCK_W); stage++) {
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
            const int mask_idx_start = 128 * stage + 16 * s + (mask_width * blockIdx.y * BLOCK_H);
            wmma::load_matrix_sync(mask_frag, mask + mask_idx_start, mask_width);
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


__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    const int W_size = ceil((float)W_out / 16);
    int m = blockIdx.x;
    int h = (blockIdx.y / W_size) * 16 + threadIdx.y;
    int w = (blockIdx.y % W_size) * 16 + threadIdx.x;
    int b = blockIdx.z;

    if (h >= H_out || w >= W_out) return;

    float acc = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                acc += in_4d(b, c, h * S + kh, w * S + kw) * mask_4d(m, c, kh, kw);
            }
        }
    }
    out_4d(b, m, h, w) = acc;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int mask_width =  ceil(C * K * K / 8.0) * 8;
    const int mask_height = ceil(M / 16.0) * 16;

    cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float));

    if (M <= 8) {
        cudaMalloc((void **)device_mask_ptr, M * C * K * K * sizeof(float));
        cudaMemcpy((void *)(*device_mask_ptr), (const void *)host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)(*device_input_ptr), (const void *)host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        cudaMalloc((void **)device_mask_ptr, mask_height * mask_width * sizeof(__half));
        __half *h_mask;
        cudaMallocHost(&h_mask, mask_height * mask_width * sizeof(__half));
        for (int i = 0; i < mask_height; i++) {
            for (int j = 0; j < mask_width; j++) {
                if (j < C * K * K && i < M) {
                    h_mask[i * mask_width + j] = __float2half(host_mask[i * C * K * K + j]);
                } else {
                    h_mask[i * mask_width + j] = static_cast<__half>(0);
                }
            }
        }
        cudaMemcpy((void *)(*device_mask_ptr), (const void *)h_mask, mask_height * mask_width * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)(*device_input_ptr), (const void *)host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
        cudaFreeHost(h_mask);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    if (M <= 8) {
        const int H_size = ceil((float)H_out / 16);
        const int W_size = ceil((float)W_out / 16);
        dim3 conv_grid(M, H_size * W_size, B);
        dim3 conv_block(16, 16, 1);
        conv_forward_kernel<<<conv_grid, conv_block>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    } else {
        dim3 mat_grid(ceil((float)(H_out * W_out) / BLOCK_W), ceil((float)M / BLOCK_H), B);
        dim3 mat_block(32, 8, 1);
        tensor_mul_kernel<<<mat_grid, mat_block>>>(device_output, device_input, (__half *)device_mask, B, M, C, H, W, K, S);
    }
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