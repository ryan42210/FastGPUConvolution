#include "conv_kernel.h"

#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>

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



__host__ void launch_conv_kernel(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
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
