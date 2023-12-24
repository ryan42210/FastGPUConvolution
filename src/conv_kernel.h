#ifndef CONV_KERNEL_H_
#define CONV_KERNEL_H_

launch_conv_kernel(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S);

#endif