# FastGPUConvolution

Part of UIUC ECE408/CS483 Final project.


An optimized convolution layer of CNN on CUDA

- GEMM implementation of convolution.
- Shared memory tiling of matrix multiplication.
- Fp16 vectorized matrix multiplication.
- Accelerating GEMM with the Tensor Cores (wmma instruction).
