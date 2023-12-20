#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "gpu_conv.h"
/*
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)

*/
constexpr int B = 500;
constexpr int M = 16;
constexpr int C = 64;
constexpr int H = 40;
constexpr int W = 40;
constexpr int K = 7;

using Vecf = std::vector<float>;

void fill_random(Vecf &input, int len) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    input.clear();
    input.assign(len, 0);
    for (auto &i : input) {
        i = dis(gen);
    }
}

Vecf get_ref_output(const Vecf &input, const Vecf &mask, int B, int M, int C, int H, int W, int K) {
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    std::vector<float> output(H_out * W_out * M * B);
    for (int b = 0; b < B; b++) {
        for (int m = 0; m < M; m++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    out_4d(b, m, h, w) = 0.0f;
                    for (int c = 0; c < C; c++) {
                        for (int kh = 0; kh < K; kh++) {
                            for (int kw = 0; kw < K; kw++) {
                                out_4d(b, m, h, w) += in_4d(b, c, h + kh, w + kw) * mask_4d(m, c, kh, kw);
                            }
                        }
                    }
                }
            }
        }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d

    return output;    
}


long long test_gpu_conv(const Vecf &input, const Vecf &mask, Vecf &output, int B, int M, int C, int H, int W, int K) {
    float *d_input, *d_mask, *d_output;

    device_mem_init(&d_input, &d_mask, &d_output, input.data(), mask.data(), B, M, C, H, W, K);
    auto start = std::chrono::steady_clock::now();
    kernel_launch(d_input, d_mask, d_output, B, M, C, H, W, K);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    device_mem_free(d_input, d_mask, d_output, output.data(), B, M, H, W);
    return elapsed_time;
}

int main() {
    Vecf h_mask;
    Vecf h_input;
    Vecf h_output;

    fill_random(h_mask, M * C * K * K);
    fill_random(h_input, C * H * W);
    auto ref_output = get_ref_output(h_input, h_mask, 1, M, C, H, W, K);
    test_gpu_conv(h_input, h_mask, h_output, 1, M, C, H, W, K);
    if (h_output != ref_output) {
        std::cerr << "Wrong convolution result." << std::endl;
    }

    fill_random(h_input, B * C * H * W);
    auto elapsed_time = test_gpu_conv(h_input, h_mask, h_output, B, M, C, H, W, K);
    std::cout << "Convolution GPU kernel takes "<< elapsed_time << " ms to finish." << std::endl;

    return 0;
}