#include <cuda_runtime_api.h>

#include "gpu_conv.h"
#include "conv_kernel.h"

void device_mem_init(float **d_input, float **d_mask, float **d_output, const float *h_input, const float *h_mask, int B, int M, int C, int H, int W, int K) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    cudaMalloc((void **)d_input, B * C * H * W * sizeof(float));
    cudaMalloc((void **)d_mask, M * C * K * K * sizeof(float));
    cudaMalloc((void **)d_output, B * M * H_out * W_out * sizeof(float));
    cudaMemcpy((void *)(*d_input), (const void *)h_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)(*d_mask), (const void *)h_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
}


void kernel_launch(float *d_input, float *d_mask, float *output, int B, int M, int C, int H, int W, int K) {

}

void device_mem_free(float *d_input, float *d_mask, float *d_output, float *h_output, int B, int M, int H, int W) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // Copy the output back to host
    cudaMemcpy(h_output, d_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
}