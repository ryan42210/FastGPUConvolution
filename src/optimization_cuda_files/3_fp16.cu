#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"

const int TILE_WIDTH = 16;

__global__ void mat_fp16_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S) {
    __shared__ __half2 tiled_mask[TILE_WIDTH * TILE_WIDTH];
    __shared__ __half tiled_unrolled[TILE_WIDTH * TILE_WIDTH];
    __half2 *vec_unrolled = (__half2 *)tiled_unrolled;

    const int b = blockIdx.z;
    
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int w = TILE_WIDTH * blockIdx.x + tx;
    const int h = TILE_WIDTH * blockIdx.y + ty;
    const int out_w = w % W_out;
    const int out_h = w / W_out;

    int h_id = ty * 2 + tx / 8;

    __half2 val2 = __float2half2_rn(0.0f);
    for (int stage = 0; stage < ceil((float)(C * K * K) / TILE_WIDTH); stage++) {
        // load mask matrix
        if (stage * TILE_WIDTH + tx < (C * K * K)) {
            if (h < M) {
                tiled_mask[ty * TILE_WIDTH + tx] = __float2half2_rn(mask[h * (C * K * K) + stage * TILE_WIDTH + tx]);
            } else {
                tiled_mask[ty * TILE_WIDTH + tx] = __float2half2_rn(0.0f);
            }
            if (h + 8 < M) {
                tiled_mask[(ty + 8) * TILE_WIDTH + tx] = __float2half2_rn(mask[(h + 8) * (C * K * K) + stage * TILE_WIDTH + tx]);
            } else {
                tiled_mask[(ty + 8) * TILE_WIDTH + tx] = __float2half2_rn(0.0f);
            }
        }

        int unrolled_h = stage * TILE_WIDTH + ty;
        if (w < H_out * W_out) {
            if (unrolled_h < (C * K * K)) {
                int c = unrolled_h / (K * K);
                int in_c = unrolled_h % (K * K);
                int in_w = (in_c % K) + out_w * S;
                int in_h = (in_c / K) + out_h * S;
                tiled_unrolled[ty * TILE_WIDTH + tx] = __float2half(input[b * (C * H * W) + c * (H * W) + in_h * (W) + in_w]);
            } else  {
                tiled_unrolled[ty * TILE_WIDTH + tx] = static_cast<__half>(0);
            }
            unrolled_h += 8;
            if (unrolled_h < (C * K * K)) {
                int c = unrolled_h / (K * K);
                int in_c = unrolled_h % (K * K);
                int in_w = (in_c % K) + out_w * S;
                int in_h = (in_c / K) + out_h * S;
                tiled_unrolled[(ty + 8) * TILE_WIDTH + tx] = __float2half(input[b * (C * H * W) + c * (H * W) + in_h * (W) + in_w]);
            } else  {
                tiled_unrolled[(ty + 8) * TILE_WIDTH + tx] = static_cast<__half>(0);
            }
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            val2 = __hfma2(tiled_mask[h_id * 16 + k], vec_unrolled[k * 8 + (tx % 8)], val2);
        }
        __syncthreads();
    }

    if (TILE_WIDTH * blockIdx.y + h_id < M) {
        int w_id = TILE_WIDTH * blockIdx.x + (tx % 8) * 2;
        if (w_id < H_out * W_out)
            output[b * (M * H_out * W_out) + (TILE_WIDTH * blockIdx.y + h_id) * (H_out * W_out) + w_id] =  __half2float(val2.x);
        if (w_id + 1 < H_out * W_out)
            output[b * (M * H_out * W_out) + (TILE_WIDTH * blockIdx.y + h_id) * (H_out * W_out) + w_id + 1] = __half2float(val2.y);
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
    dim3 mat_block(16, 8, 1);

    mat_fp16_kernel<<<mat_grid, mat_block>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
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


// Running test case 1
// B = 1 M = 3 C = 3 H = 224 W = 224 K = 3 S = 1
// Running test case 2
// B = 2 M = 3 C = 3 H = 301 W = 301 K = 3 S = 2
// Running test case 3
// B = 3 M = 3 C = 3 H = 196 W = 196 K = 3 S = 3
// Running test case 4
// B = 4 M = 3 C = 3 H = 239 W = 239 K = 3 S = 4
// All test cases passed
// Test batch size: 5000
// Loading fashion-mnist data...Done
// Loading model...Done
// Conv-GPU==
// Layer Time: 342.293 ms
// Op Time: 22.0901 ms
// Conv-GPU==
// Layer Time: 244.418 ms
// Op Time: 14.1458 ms

// Test Accuracy: 0.8712


// real    0m51.117s
// user    0m50.103s
// sys     0m0.936s