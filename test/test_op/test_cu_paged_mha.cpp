// #include <gtest/gtest.h>
// #include <tensor/tensor.h>
// #include <base/alloc.h>
// #include <base/cuda_config.h>
// #include <cuda_runtime_api.h>
// #include "../source/op/kernels/cuda/mha_kernel.cuh"

// TEST(TEST_MHA, test_paged_attention_mapping) {
//   // 1. 设置极简的测试维度
//   int32_t head_num = 2;
//   int32_t head_size = 64;
//   int32_t kv_dim = head_num * head_size; // 128
//   int32_t kv_mul = 1;
//   int32_t block_size = 2;
//   int32_t seq_len = 4;
//   int32_t pos = 3;
//   int32_t layer_index = 0;

//   auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
//   auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

//   // 2. 初始化输入输出 Tensor
//   tensor::Tensor query(base::DataType::kDataTypeFp32, head_num * head_size, true, alloc_cu);
//   tensor::Tensor score(base::DataType::kDataTypeFp32, head_num, seq_len, true, alloc_cu);
//   tensor::Tensor out(base::DataType::kDataTypeFp32, head_num * head_size, true, alloc_cu);

//   // 3. 初始化全局物理 KV Cache (分配具有10个物理块的池子)
//   int32_t num_physical_blocks = 10;
//   tensor::Tensor key_cache(base::DataType::kDataTypeFp32, 1, num_physical_blocks * block_size, kv_dim, true, alloc_cu);
//   tensor::Tensor value_cache(base::DataType::kDataTypeFp32, 1, num_physical_blocks * block_size, kv_dim, true, alloc_cu);

//   // 4. 构造 Paged Attention 乱序 Block Table
//   tensor::Tensor block_table(base::DataType::kDataTypeInt32, seq_len / block_size, true, alloc_cu);
//   int32_t host_block_table[] = {5, 2};
//   cudaMemcpy(const_cast<int32_t*>(block_table.ptr<int32_t>()), host_block_table, 2 * sizeof(int32_t), cudaMemcpyHostToDevice);

//   // 5. 准备测试数据：将需要的 QKV初始化为1.0f
//   float* host_q = new float[head_num * head_size];
//   for (int i = 0; i < head_num * head_size; i++) {
//     host_q[i] = 1.0f;
//   }
//   cudaMemcpy(const_cast<float*>(query.ptr<float>()), host_q, head_num * head_size * sizeof(float), cudaMemcpyHostToDevice);

//   // 初始化整个物理池为 0.0f
//   cudaMemset(const_cast<float*>(key_cache.ptr<float>()), 0, key_cache.size() * sizeof(float));
//   cudaMemset(const_cast<float*>(value_cache.ptr<float>()), 0, value_cache.size() * sizeof(float));

//   // 仅在物理块 5 和 2 中写入1.0f， 其余块保持为 0.0f
//   float* host_kv_block = new float[block_size * kv_dim];
//   for (int i = 0; i < block_size * kv_dim; i++) {
//     host_kv_block[i] = 1.0f;
//   }
//   float* d_k = const_cast<float*>(key_cache.ptr<float>());
//   float* d_v = const_cast<float*>(value_cache.ptr<float>());

//   // 写入块 5
//   cudaMemcpy(d_k + 5 * block_size * kv_dim, host_kv_block, block_size * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_v + 5 * block_size * kv_dim, host_kv_block, block_size * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

//   // 写入块 2 
//   cudaMemcpy(d_k + 2 * block_size * kv_dim, host_kv_block, block_size * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_v + 2 * block_size * kv_dim, host_kv_block, block_size * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

//   // 6. 启动 Kernel
//   kernel::CudaConfig config;
//   cudaStreamCreate(&config.stream);

//   kernel::mha_kernel_cu(pos, head_num, layer_index, seq_len, kv_dim, kv_mul, head_size, 
//                         out, query, score, key_cache, value_cache, block_table, block_size,  
//                         base::DeviceType::kDeviceCUDA, &config);
//   cudaStreamSynchronize(config.stream);
  
//   // 7. 验证结果
//   float* host_out = new float[head_num * head_size];
//   cudaMemcpy(host_out, const_cast<float*>(out.ptr<float>()), head_num * head_size * sizeof(float), cudaMemcpyDeviceToHost);

//   // 数学推导验证：由于所有参与计算的 Q, K, V 均为 1.0f
//   // Softmax 归一化后权重均等，加权求和 V 之后，结果必定全为 1.0f
//   // 如果读取到了错误的物理块(值为0)，最终结果就会小于 1.0f
//   for (int i = 0; i < head_num * head_size; ++i) {
//     ASSERT_NEAR(host_out[i], 1.0f, 1e-4);
//   }

//   delete[] host_q;
//   delete[] host_kv_block;
//   delete[] host_out;
//   cudaStreamDestroy(config.stream);
// }


#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include <base/alloc.h>
#include <base/cuda_config.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "../source/op/kernels/cuda/mha_kernel.cuh"

// 宏定义：用于精准捕捉任何 CUDA 运行时的隐藏报错
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " (" << cudaGetErrorString(err) << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

TEST(TEST_MHA, test_paged_attention_mapping) {
  std::cout << "[DEBUG] 1. 初始化 CUDA 设备..." << std::endl;
  // 必须显式设置 Device，否则 Allocator 可能会分配失败导致后续 ptr() 为 nullptr
  CUDA_CHECK(cudaSetDevice(0));

  int32_t head_num = 2;
  int32_t head_size = 64;
  int32_t kv_dim = head_num * head_size; // 128
  int32_t kv_mul = 1;
  int32_t block_size = 2;
  int32_t seq_len = 4;
  int32_t pos = 3;
  int32_t layer_index = 0;

  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  ASSERT_NE(alloc_cu, nullptr) << "CUDA Allocator 初始化失败！";

  std::cout << "[DEBUG] 2. 分配 Tensor 显存..." << std::endl;
  tensor::Tensor query(base::DataType::kDataTypeFp32, head_num * head_size, true, alloc_cu);
  tensor::Tensor score(base::DataType::kDataTypeFp32, head_num, seq_len, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, head_num * head_size, true, alloc_cu);

  int32_t num_physical_blocks = 10;
  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, 1, num_physical_blocks * block_size, kv_dim, true, alloc_cu);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, 1, num_physical_blocks * block_size, kv_dim, true, alloc_cu);
  tensor::Tensor block_table(base::DataType::kDataTypeInt32, seq_len / block_size, true, alloc_cu);

  // 检查是否所有 Tensor 都成功在 GPU 上分配到了显存
  ASSERT_NE(query.ptr<float>(), nullptr) << "Query Tensor 分配失败！";
  ASSERT_NE(key_cache.ptr<float>(), nullptr) << "Key Cache 分配失败！";
  ASSERT_NE(block_table.ptr<int32_t>(), nullptr) << "Block Table 分配失败！";

  std::cout << "[DEBUG] 3. 拷贝数据到 GPU..." << std::endl;
  int32_t host_block_table[] = {5, 2};
  CUDA_CHECK(cudaMemcpy(const_cast<int32_t*>(block_table.ptr<int32_t>()), host_block_table, 2 * sizeof(int32_t), cudaMemcpyHostToDevice));

  float* host_q = new float[head_num * head_size];
  for (int i = 0; i < head_num * head_size; ++i) host_q[i] = 1.0f;
  CUDA_CHECK(cudaMemcpy(const_cast<float*>(query.ptr<float>()), host_q, head_num * head_size * sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemset(const_cast<float*>(key_cache.ptr<float>()), 0, key_cache.size() * sizeof(float)));
  CUDA_CHECK(cudaMemset(const_cast<float*>(value_cache.ptr<float>()), 0, value_cache.size() * sizeof(float)));

  float* host_kv_block = new float[block_size * kv_dim];
  for (int i = 0; i < block_size * kv_dim; ++i) host_kv_block[i] = 1.0f;

  float* d_k = const_cast<float*>(key_cache.ptr<float>());
  float* d_v = const_cast<float*>(value_cache.ptr<float>());

  // 写入块 5 和块 2
  CUDA_CHECK(cudaMemcpy(d_k + 5 * block_size * kv_dim, host_kv_block, block_size * kv_dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v + 5 * block_size * kv_dim, host_kv_block, block_size * kv_dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k + 2 * block_size * kv_dim, host_kv_block, block_size * kv_dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v + 2 * block_size * kv_dim, host_kv_block, block_size * kv_dim * sizeof(float), cudaMemcpyHostToDevice));

  std::cout << "[DEBUG] 4. 启动 GPU Kernel..." << std::endl;
  kernel::CudaConfig config;
  CUDA_CHECK(cudaStreamCreate(&config.stream));

  kernel::mha_kernel_cu(pos, head_num, layer_index, seq_len, kv_dim, kv_mul, head_size, 
                        out, query, score, key_cache, value_cache, block_table, block_size,  
                        base::DeviceType::kDeviceCUDA, &config);
                        
  // 同步并捕获 Kernel 内部可能发生的任何指针越界（非法内存访问）错误
  CUDA_CHECK(cudaStreamSynchronize(config.stream));
  
  std::cout << "[DEBUG] 5. 校验计算结果..." << std::endl;
  float* host_out = new float[head_num * head_size];
  CUDA_CHECK(cudaMemcpy(host_out, const_cast<float*>(out.ptr<float>()), head_num * head_size * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < head_num * head_size; ++i) {
    ASSERT_NEAR(host_out[i], 1.0f, 1e-4);
  }

  std::cout << "[DEBUG] 测试圆满通过！" << std::endl;
  delete[] host_q;
  delete[] host_kv_block;
  delete[] host_out;
}