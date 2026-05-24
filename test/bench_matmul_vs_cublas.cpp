#include <iostream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <cmath>
#include <glog/logging.h>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "base/alloc.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"
#include "op/kernels/cuda/matmul_kernel.cuh"

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::cerr << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__               \
                << " code=" << err__ << " msg=" << cudaGetErrorString(err__)    \
                << std::endl;                                                    \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                           \
    cublasStatus_t st__ = (call);                                                \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                         \
      std::cerr << "[CUBLAS ERROR] " << __FILE__ << ":" << __LINE__             \
                << " status=" << static_cast<int>(st__) << std::endl;            \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

static float run_custom_matmul_ms(const tensor::Tensor& input,
                                  const tensor::Tensor& weight,
                                  const tensor::Tensor& output,
                                  kernel::CudaConfig* cfg,
                                  int warmup,
                                  int iters) {
  for (int i = 0; i < warmup; ++i) {
    kernel::matmul_kernel_cu(input, weight, output, 1.0f, cfg);
  }
  CUDA_CHECK(cudaStreamSynchronize(cfg->stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, cfg->stream));
  for (int i = 0; i < iters; ++i) {
    kernel::matmul_kernel_cu(input, weight, output, 1.0f, cfg);
  }
  CUDA_CHECK(cudaEventRecord(stop, cfg->stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

static float run_cublas_matmul_ms(const tensor::Tensor& input,
                                  const tensor::Tensor& weight,
                                  const tensor::Tensor& output,
                                  cublasHandle_t handle,
                                  cudaStream_t stream,
                                  int B, int K, int M,
                                  int warmup,
                                  int iters) {
  // 目标: C_row(B, K) = A_row(B, M) * W_row(K, M)^T
  // 利用列主序等价：调用 cublasSgemm(op(W)=T, op(A)=N) 得到 C_col(K, B)
  // 参数:
  //   W_col: 视作 (M x K), lda=M
  //   A_col: 视作 (M x B), ldb=M
  //   C_col: (K x B), ldc=K
  const float alpha = 1.0f;
  const float beta  = 0.0f;

  const float* A = input.ptr<float>();
  const float* W = weight.ptr<float>();
  float* C = const_cast<float*>(output.ptr<float>());

  for (int i = 0; i < warmup; ++i) {
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             K, B, M,
                             &alpha,
                             W, M,
                             A, M,
                             &beta,
                             C, K));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             K, B, M,
                             &alpha,
                             W, M,
                             A, M,
                             &beta,
                             C, K));
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

int main() {
  CUDA_CHECK(cudaSetDevice(0));

  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  if (!alloc_cu) {
    std::cerr << "Failed to get CUDA allocator." << std::endl;
    return 1;
  }

  kernel::CudaConfig cfg{};
  CUDA_CHECK(cudaStreamCreate(&cfg.stream));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, cfg.stream));

  // (B, K, M): C[B,K] = A[B,M] * W[K,M]^T
  std::vector<std::tuple<int,int,int>> shapes = {
      {1,   768,  768},   // attention qkv-like
      {20,  768,  768},   // 你的并发场景
      {20, 3072,  768},   // FFN W1/W3-like
      {20,  768, 3072},   // FFN W2-like
      {64, 3072,  768},
  };

  const int warmup = 100;
  const int iters  = 500;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "B,K,M,custom_ms,cublas_ms,custom_tflops,cublas_tflops,custom/cublas\n";

  for (auto [B, K, M] : shapes) {
    tensor::Tensor input(base::DataType::kDataTypeFp32, B, M, true, alloc_cu);
    tensor::Tensor weight(base::DataType::kDataTypeFp32, K, M, true, alloc_cu);
    tensor::Tensor out_custom(base::DataType::kDataTypeFp32, B, K, true, alloc_cu);
    tensor::Tensor out_cublas(base::DataType::kDataTypeFp32, B, K, true, alloc_cu);

    // 避免H2D影响，直接在GPU初始化
    CUDA_CHECK(cudaMemsetAsync(const_cast<float*>(input.ptr<float>()),  0, input.size()  * sizeof(float), cfg.stream));
    CUDA_CHECK(cudaMemsetAsync(const_cast<float*>(weight.ptr<float>()), 0, weight.size() * sizeof(float), cfg.stream));
    CUDA_CHECK(cudaMemsetAsync(const_cast<float*>(out_custom.ptr<float>()), 0, out_custom.size() * sizeof(float), cfg.stream));
    CUDA_CHECK(cudaMemsetAsync(const_cast<float*>(out_cublas.ptr<float>()), 0, out_cublas.size() * sizeof(float), cfg.stream));
    CUDA_CHECK(cudaStreamSynchronize(cfg.stream));

    float custom_ms = run_custom_matmul_ms(input, weight, out_custom, &cfg, warmup, iters);
    float cublas_ms = run_cublas_matmul_ms(input, weight, out_cublas, handle, cfg.stream, B, K, M, warmup, iters);

    double flops = 2.0 * static_cast<double>(B) * K * M;
    double custom_tflops = (flops / (custom_ms * 1e-3)) / 1e12;
    double cublas_tflops = (flops / (cublas_ms * 1e-3)) / 1e12;
    double ratio = custom_ms / cublas_ms; // >1 表示你更慢

    std::cout << B << "," << K << "," << M << ","
              << custom_ms << "," << cublas_ms << ","
              << custom_tflops << "," << cublas_tflops << ","
              << ratio << "\n";
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(cfg.stream));
  return 0;
}