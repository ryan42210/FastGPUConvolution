#ifndef GPU_CONV_H_
#define GPU_CONV_H_

void device_mem_init(float **d_input, float **d_mask, float **d_output, const float *h_input, const float *h_mask, int B, int M, int C, int H, int W, int K);
void kernel_launch(float *d_input, float *d_mask, float *output, int B, int M, int C, int H, int W, int K);
void device_mem_free(float *d_input, float *d_mask, float *d_output, float *h_output, int B, int M, int H, int W);

#endif