.
├── benchmark_frag
├── benchmark_fragmentation.cpp
├── build
│   ├── bench
│   ├── CMakeCache.txt
│   ├── CMakeFiles
│   │   ├── 3.22.1
│   │   │   ├── CMakeCUDACompiler.cmake
│   │   │   ├── CMakeCXXCompiler.cmake
│   │   │   ├── CMakeDetermineCompilerABI_CUDA.bin
│   │   │   ├── CMakeDetermineCompilerABI_CXX.bin
│   │   │   ├── CMakeSystem.cmake
│   │   │   ├── CompilerIdCUDA
│   │   │   │   ├── a.out
│   │   │   │   ├── CMakeCUDACompilerId.cu
│   │   │   │   └── tmp
│   │   │   │       ├── a_dlink.fatbin
│   │   │   │       ├── a_dlink.fatbin.c
│   │   │   │       ├── a_dlink.o
│   │   │   │       ├── a_dlink.reg.c
│   │   │   │       ├── a_dlink.sm_52.cubin
│   │   │   │       ├── CMakeCUDACompilerId.cpp1.ii
│   │   │   │       ├── CMakeCUDACompilerId.cpp4.ii
│   │   │   │       ├── CMakeCUDACompilerId.cudafe1.c
│   │   │   │       ├── CMakeCUDACompilerId.cudafe1.cpp
│   │   │   │       ├── CMakeCUDACompilerId.cudafe1.gpu
│   │   │   │       ├── CMakeCUDACompilerId.cudafe1.stub.c
│   │   │   │       ├── CMakeCUDACompilerId.fatbin
│   │   │   │       ├── CMakeCUDACompilerId.fatbin.c
│   │   │   │       ├── CMakeCUDACompilerId.module_id
│   │   │   │       ├── CMakeCUDACompilerId.o
│   │   │   │       ├── CMakeCUDACompilerId.ptx
│   │   │   │       └── CMakeCUDACompilerId.sm_52.cubin
│   │   │   └── CompilerIdCXX
│   │   │       ├── a.out
│   │   │       ├── CMakeCXXCompilerId.cpp
│   │   │       └── tmp
│   │   ├── cmake.check_cache
│   │   ├── CMakeDirectoryInformation.cmake
│   │   ├── CMakeOutput.log
│   │   ├── CMakeTmp
│   │   ├── llama.dir
│   │   │   ├── build.make
│   │   │   ├── cmake_clean.cmake
│   │   │   ├── cmake_device_link.o
│   │   │   ├── compiler_depend.internal
│   │   │   ├── compiler_depend.make
│   │   │   ├── compiler_depend.ts
│   │   │   ├── DependInfo.cmake
│   │   │   ├── depend.make
│   │   │   ├── dlink.txt
│   │   │   ├── flags.make
│   │   │   ├── kuiper
│   │   │   │   └── source
│   │   │   │       ├── base
│   │   │   │       │   ├── alloc.cpp.o
│   │   │   │       │   ├── alloc.cpp.o.d
│   │   │   │       │   ├── alloc_cpu.cpp.o
│   │   │   │       │   ├── alloc_cpu.cpp.o.d
│   │   │   │       │   ├── alloc_cu.cpp.o
│   │   │   │       │   ├── alloc_cu.cpp.o.d
│   │   │   │       │   ├── base.cpp.o
│   │   │   │       │   ├── base.cpp.o.d
│   │   │   │       │   ├── buffer.cpp.o
│   │   │   │       │   ├── buffer.cpp.o.d
│   │   │   │       │   ├── unicode.cpp.o
│   │   │   │       │   ├── unicode.cpp.o.d
│   │   │   │       │   ├── unicode-data.cpp.o
│   │   │   │       │   └── unicode-data.cpp.o.d
│   │   │   │       ├── model
│   │   │   │       │   ├── llama3.cpp.o
│   │   │   │       │   ├── llama3.cpp.o.d
│   │   │   │       │   ├── model.cpp.o
│   │   │   │       │   ├── model.cpp.o.d
│   │   │   │       │   ├── qwen2.cpp.o
│   │   │   │       │   ├── qwen2.cpp.o.d
│   │   │   │       │   ├── qwen3.cpp.o
│   │   │   │       │   ├── qwen3.cpp.o.d
│   │   │   │       │   ├── raw_model_data.cpp.o
│   │   │   │       │   └── raw_model_data.cpp.o.d
│   │   │   │       ├── op
│   │   │   │       │   ├── add.cpp.o
│   │   │   │       │   ├── add.cpp.o.d
│   │   │   │       │   ├── embedding.cpp.o
│   │   │   │       │   ├── embedding.cpp.o.d
│   │   │   │       │   ├── encode.cpp.o
│   │   │   │       │   ├── encode.cpp.o.d
│   │   │   │       │   ├── kernels
│   │   │   │       │   │   ├── cpu
│   │   │   │       │   │   │   ├── add_kernel.cpp.o
│   │   │   │       │   │   │   ├── add_kernel.cpp.o.d
│   │   │   │       │   │   │   ├── emb_kernel.cpp.o
│   │   │   │       │   │   │   ├── emb_kernel.cpp.o.d
│   │   │   │       │   │   │   ├── matmul_kernel.cpp.o
│   │   │   │       │   │   │   ├── matmul_kernel.cpp.o.d
│   │   │   │       │   │   │   ├── mha_kernel.cpp.o
│   │   │   │       │   │   │   ├── mha_kernel.cpp.o.d
│   │   │   │       │   │   │   ├── rmsnorm_kernel.cpp.o
│   │   │   │       │   │   │   ├── rmsnorm_kernel.cpp.o.d
│   │   │   │       │   │   │   ├── rope_kernel.cpp.o
│   │   │   │       │   │   │   ├── rope_kernel.cpp.o.d
│   │   │   │       │   │   │   ├── scale_kernel.cpp.o
│   │   │   │       │   │   │   ├── scale_kernel.cpp.o.d
│   │   │   │       │   │   │   ├── scale_sum_kernel.cpp.o
│   │   │   │       │   │   │   ├── scale_sum_kernel.cpp.o.d
│   │   │   │       │   │   │   ├── softmax_kernel.cpp.o
│   │   │   │       │   │   │   ├── softmax_kernel.cpp.o.d
│   │   │   │       │   │   │   ├── swiglu_kernel.cpp.o
│   │   │   │       │   │   │   └── swiglu_kernel.cpp.o.d
│   │   │   │       │   │   ├── cuda
│   │   │   │       │   │   │   ├── add_kernel.cu.o
│   │   │   │       │   │   │   ├── add_kernel.cu.o.d
│   │   │   │       │   │   │   ├── argmax_kernel.cu.o
│   │   │   │       │   │   │   ├── argmax_kernel.cu.o.d
│   │   │   │       │   │   │   ├── emb_kernel.cu.o
│   │   │   │       │   │   │   ├── emb_kernel.cu.o.d
│   │   │   │       │   │   │   ├── kvcache_kernel.cu.o
│   │   │   │       │   │   │   ├── kvcache_kernel.cu.o.d
│   │   │   │       │   │   │   ├── matmul_kernel.cu.o
│   │   │   │       │   │   │   ├── matmul_kernel.cu.o.d
│   │   │   │       │   │   │   ├── mha_kernel.cu.o
│   │   │   │       │   │   │   ├── mha_kernel.cu.o.d
│   │   │   │       │   │   │   ├── rmsnorm_kernel.cu.o
│   │   │   │       │   │   │   ├── rmsnorm_kernel.cu.o.d
│   │   │   │       │   │   │   ├── rope_kernel.cu.o
│   │   │   │       │   │   │   ├── rope_kernel.cu.o.d
│   │   │   │       │   │   │   ├── swiglu_kernel.cu.o
│   │   │   │       │   │   │   └── swiglu_kernel.cu.o.d
│   │   │   │       │   │   ├── kernels_interfaces.cpp.o
│   │   │   │       │   │   └── kernels_interfaces.cpp.o.d
│   │   │   │       │   ├── layer.cpp.o
│   │   │   │       │   ├── layer.cpp.o.d
│   │   │   │       │   ├── matmul.cpp.o
│   │   │   │       │   ├── matmul.cpp.o.d
│   │   │   │       │   ├── mha.cpp.o
│   │   │   │       │   ├── mha.cpp.o.d
│   │   │   │       │   ├── rmsnorm.cpp.o
│   │   │   │       │   ├── rmsnorm.cpp.o.d
│   │   │   │       │   ├── rope.cpp.o
│   │   │   │       │   ├── rope.cpp.o.d
│   │   │   │       │   ├── swiglu.cpp.o
│   │   │   │       │   └── swiglu.cpp.o.d
│   │   │   │       ├── sampler
│   │   │   │       │   ├── argmax_sampler.cpp.o
│   │   │   │       │   └── argmax_sampler.cpp.o.d
│   │   │   │       └── tensor
│   │   │   │           ├── tensor.cpp.o
│   │   │   │           └── tensor.cpp.o.d
│   │   │   ├── link.txt
│   │   │   └── progress.make
│   │   ├── Makefile2
│   │   ├── Makefile.cmake
│   │   ├── progress.marks
│   │   └── TargetDirectories.txt
│   ├── cmake_install.cmake
│   ├── compile_commands.json
│   ├── demo
│   │   ├── CMakeFiles
│   │   │   ├── CMakeDirectoryInformation.cmake
│   │   │   ├── llama_infer.dir
│   │   │   │   ├── build.make
│   │   │   │   ├── cmake_clean.cmake
│   │   │   │   ├── compiler_depend.internal
│   │   │   │   ├── compiler_depend.make
│   │   │   │   ├── compiler_depend.ts
│   │   │   │   ├── DependInfo.cmake
│   │   │   │   ├── depend.make
│   │   │   │   ├── flags.make
│   │   │   │   ├── link.txt
│   │   │   │   ├── main.cpp.o
│   │   │   │   ├── main.cpp.o.d
│   │   │   │   └── progress.make
│   │   │   └── progress.marks
│   │   ├── cmake_install.cmake
│   │   ├── llama_infer
│   │   └── Makefile
│   ├── detect_cuda_compute_capabilities.cu
│   ├── Makefile
│   └── test
│       ├── bench_matmul_vs_cublas
│       ├── CMakeFiles
│       │   ├── bench_matmul_vs_cublas.dir
│       │   │   ├── bench_matmul_vs_cublas.cpp.o
│       │   │   ├── bench_matmul_vs_cublas.cpp.o.d
│       │   │   ├── build.make
│       │   │   ├── cmake_clean.cmake
│       │   │   ├── compiler_depend.internal
│       │   │   ├── compiler_depend.make
│       │   │   ├── compiler_depend.ts
│       │   │   ├── DependInfo.cmake
│       │   │   ├── depend.make
│       │   │   ├── flags.make
│       │   │   ├── link.txt
│       │   │   └── progress.make
│       │   ├── CMakeDirectoryInformation.cmake
│       │   ├── progress.marks
│       │   └── test_llm.dir
│       │       ├── bench_matmul_vs_cublas.cpp.o
│       │       ├── bench_matmul_vs_cublas.cpp.o.d
│       │       ├── build.make
│       │       ├── cmake_clean.cmake
│       │       ├── cmake_device_link.o
│       │       ├── compiler_depend.internal
│       │       ├── compiler_depend.make
│       │       ├── compiler_depend.ts
│       │       ├── DependInfo.cmake
│       │       ├── depend.make
│       │       ├── dlink.txt
│       │       ├── flags.make
│       │       ├── link.txt
│       │       ├── optimized
│       │       │   ├── test_allocator.cpp.o
│       │       │   └── test_allocator.cpp.o.d
│       │       ├── progress.make
│       │       ├── test_cu
│       │       │   ├── test_cu_wrap.cpp.o
│       │       │   └── test_cu_wrap.cpp.o.d
│       │       ├── test_main.cpp.o
│       │       ├── test_main.cpp.o.d
│       │       ├── test_model
│       │       │   ├── test_llama_cpu.cpp.o
│       │       │   └── test_llama_cpu.cpp.o.d
│       │       ├── test_op
│       │       │   ├── test_cu_add.cpp.o
│       │       │   ├── test_cu_add.cpp.o.d
│       │       │   ├── test_cu_emb.cpp.o
│       │       │   ├── test_cu_emb.cpp.o.d
│       │       │   ├── test_cu_matmul.cpp.o
│       │       │   ├── test_cu_matmul.cpp.o.d
│       │       │   ├── test_cu_paged_mha.cpp.o
│       │       │   ├── test_cu_paged_mha.cpp.o.d
│       │       │   ├── test_cu_rmsnorm.cpp.o
│       │       │   ├── test_cu_rmsnorm.cpp.o.d
│       │       │   ├── test_cu_rope.cpp.o
│       │       │   ├── test_cu_rope.cpp.o.d
│       │       │   ├── test_cu_scale.cpp.o
│       │       │   ├── test_cu_scale.cpp.o.d
│       │       │   ├── test_cu_softmax.cpp.o
│       │       │   ├── test_cu_softmax.cpp.o.d
│       │       │   ├── test_cu_swiglu.cpp.o
│       │       │   ├── test_cu_swiglu.cpp.o.d
│       │       │   ├── test_load.cpp.o
│       │       │   └── test_load.cpp.o.d
│       │       ├── test_tensor
│       │       │   ├── test_buffer.cpp.o
│       │       │   ├── test_buffer.cpp.o.d
│       │       │   ├── test_tensor.cpp.o
│       │       │   └── test_tensor.cpp.o.d
│       │       ├── utils.cu.o
│       │       └── utils.cu.o.d
│       ├── cmake_install.cmake
│       ├── Makefile
│       └── test_llm
├── cmake
│   ├── CPM.cmake
│   └── cuda.cmake
├── CMakeLists.txt
├── demo
│   ├── build
│   │   └── Debug
│   ├── chat_qwen.cpp
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── main_old.cpp
│   ├── main_qwen3.cpp
│   ├── main_qwen.cpp
│   ├── test.cpp
│   ├── test_prompt.cpp
│   └── test_tokens.cpp
├── dockerfile
├── hf_infer
│   ├── llama3_infer.py
│   └── qwen2_infer.py
├── imgs
│   ├── do.gif
│   ├── me.jpg
│   ├── mulu.jpg
│   └── qa.jpg
├── kuiper
│   ├── include
│   │   ├── base
│   │   │   ├── alloc.h
│   │   │   ├── base.h
│   │   │   ├── block_manager.h
│   │   │   ├── buffer.h
│   │   │   ├── cuda_config.h
│   │   │   ├── tick.h
│   │   │   ├── tiktoken.h
│   │   │   ├── unicode-data.h
│   │   │   ├── unicode.h
│   │   │   └── unordered_dense.h
│   │   ├── model
│   │   │   ├── config.h
│   │   │   ├── kv_cache_manager.h
│   │   │   ├── llama3.h
│   │   │   ├── model.h
│   │   │   ├── qwen2.h
│   │   │   ├── qwen3.h
│   │   │   └── raw_model_data.h
│   │   ├── op
│   │   │   ├── add.h
│   │   │   ├── embedding.h
│   │   │   ├── encode.h
│   │   │   ├── layer.h
│   │   │   ├── matmul.h
│   │   │   ├── mha.h
│   │   │   ├── rmsnorm.h
│   │   │   ├── rope.h
│   │   │   └── swiglu.h
│   │   ├── sampler
│   │   │   ├── argmax_sampler.h
│   │   │   └── sampler.h
│   │   └── tensor
│   │       └── tensor.h
│   └── source
│       ├── base
│       │   ├── alloc.cpp
│       │   ├── alloc_cpu.cpp
│       │   ├── alloc_cu.cpp
│       │   ├── base.cpp
│       │   ├── buffer.cpp
│       │   ├── unicode.cpp
│       │   └── unicode-data.cpp
│       ├── model
│       │   ├── llama3.cpp
│       │   ├── model.cpp
│       │   ├── qwen2.cpp
│       │   ├── qwen3.cpp
│       │   └── raw_model_data.cpp
│       ├── op
│       │   ├── add.cpp
│       │   ├── embedding.cpp
│       │   ├── encode.cpp
│       │   ├── kernels
│       │   │   ├── cpu
│       │   │   │   ├── add_kernel.cpp
│       │   │   │   ├── add_kernel.h
│       │   │   │   ├── emb_kernel.cpp
│       │   │   │   ├── emb_kernel.h
│       │   │   │   ├── matmul_kernel.cpp
│       │   │   │   ├── matmul_kernel.h
│       │   │   │   ├── mha_kernel.cpp
│       │   │   │   ├── mha_kernel.h
│       │   │   │   ├── rmsnorm_kernel.cpp
│       │   │   │   ├── rmsnorm_kernel.h
│       │   │   │   ├── rope_kernel.cpp
│       │   │   │   ├── rope_kernel.h
│       │   │   │   ├── scale_kernel.cpp
│       │   │   │   ├── scale_kernel.h
│       │   │   │   ├── scale_sum_kernel.cpp
│       │   │   │   ├── scale_sum_kernel.h
│       │   │   │   ├── softmax_kernel.cpp
│       │   │   │   ├── softmax_kernel.h
│       │   │   │   ├── swiglu_kernel.cpp
│       │   │   │   └── swiglu_kernel.h
│       │   │   ├── cuda
│       │   │   │   ├── add_kernel.cu
│       │   │   │   ├── add_kernel.cuh
│       │   │   │   ├── argmax_kernel.cu
│       │   │   │   ├── argmax_kernel.cuh
│       │   │   │   ├── emb_kernel.cu
│       │   │   │   ├── emb_kernel.cuh
│       │   │   │   ├── kvcache_kernel.cu
│       │   │   │   ├── kvcache_kernel.cuh
│       │   │   │   ├── matmul_kernel.cu
│       │   │   │   ├── matmul_kernel.cuh
│       │   │   │   ├── mha_kernel.cu
│       │   │   │   ├── mha_kernel.cuh
│       │   │   │   ├── rmsnorm_kernel.cu
│       │   │   │   ├── rmsnorm_kernel.cuh
│       │   │   │   ├── rope_kernel.cu
│       │   │   │   ├── rope_kernel.cuh
│       │   │   │   ├── swiglu_kernel.cu
│       │   │   │   └── swiglu_kernel.cuh
│       │   │   ├── kernels_interface.h
│       │   │   └── kernels_interfaces.cpp
│       │   ├── layer.cpp
│       │   ├── matmul.cpp
│       │   ├── mha.cpp
│       │   ├── rmsnorm.cpp
│       │   ├── rope.cpp
│       │   └── swiglu.cpp
│       ├── sampler
│       │   └── argmax_sampler.cpp
│       └── tensor
│           └── tensor.cpp
├── lib
│   └── libllama.so
├── logs
│   ├── llama_infer.ERROR -> llama_infer.keqing.wyk.log.ERROR.20260424-221904.2672540
│   ├── llama_infer.FATAL -> llama_infer.keqing.wyk.log.FATAL.20260424-221904.2672540
│   ├── llama_infer.INFO -> llama_infer.keqing.wyk.log.INFO.20260426-115621.2766961
│   ├── llama_infer.keqing.wyk.log.ERROR.20260424-190124.2646288
│   ├── llama_infer.keqing.wyk.log.ERROR.20260424-190225.2646769
│   ├── llama_infer.keqing.wyk.log.ERROR.20260424-204851.2657945
│   ├── llama_infer.keqing.wyk.log.ERROR.20260424-210019.2659464
│   ├── llama_infer.keqing.wyk.log.ERROR.20260424-214021.2664058
│   ├── llama_infer.keqing.wyk.log.ERROR.20260424-214918.2665829
│   ├── llama_infer.keqing.wyk.log.ERROR.20260424-221316.2671236
│   ├── llama_infer.keqing.wyk.log.ERROR.20260424-221904.2672540
│   ├── llama_infer.keqing.wyk.log.FATAL.20260424-190124.2646288
│   ├── llama_infer.keqing.wyk.log.FATAL.20260424-190225.2646769
│   ├── llama_infer.keqing.wyk.log.FATAL.20260424-204851.2657945
│   ├── llama_infer.keqing.wyk.log.FATAL.20260424-210019.2659464
│   ├── llama_infer.keqing.wyk.log.FATAL.20260424-214021.2664058
│   ├── llama_infer.keqing.wyk.log.FATAL.20260424-214918.2665829
│   ├── llama_infer.keqing.wyk.log.FATAL.20260424-221316.2671236
│   ├── llama_infer.keqing.wyk.log.FATAL.20260424-221904.2672540
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-003506.2550819
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-003848.2553198
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-103726.2600389
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-104529.2601617
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-105250.2603003
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-152223.2616334
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-160011.2621049
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-160354.2621661
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-160526.2622039
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-161038.2622703
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-161312.2623320
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-161713.2624051
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-162829.2625464
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-162848.2625617
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-164409.2627974
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-164712.2628720
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-164743.2628977
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-164757.2629109
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-164823.2629356
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-164838.2629497
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-165004.2630024
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-165337.2630769
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-165406.2631033
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-165452.2631286
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-165753.2631736
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-172141.2634654
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-172147.2634788
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-172535.2635819
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-175443.2639059
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-175831.2639622
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-190102.2646057
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-190124.2646288
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-190225.2646769
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-204851.2657945
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-210019.2659464
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-214021.2664058
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-214918.2665829
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-221316.2671236
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-221904.2672540
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-223452.2675051
│   ├── llama_infer.keqing.wyk.log.INFO.20260424-224721.2676784
│   ├── llama_infer.keqing.wyk.log.INFO.20260425-180124.2706265
│   ├── llama_infer.keqing.wyk.log.INFO.20260425-180506.2706961
│   ├── llama_infer.keqing.wyk.log.INFO.20260425-182122.2709217
│   ├── llama_infer.keqing.wyk.log.INFO.20260425-204933.2723766
│   ├── llama_infer.keqing.wyk.log.INFO.20260425-214426.2730678
│   ├── llama_infer.keqing.wyk.log.INFO.20260426-094829.2750409
│   ├── llama_infer.keqing.wyk.log.INFO.20260426-112922.2762429
│   ├── llama_infer.keqing.wyk.log.INFO.20260426-113157.2762992
│   ├── llama_infer.keqing.wyk.log.INFO.20260426-114006.2764556
│   ├── llama_infer.keqing.wyk.log.INFO.20260426-114123.2764871
│   ├── llama_infer.keqing.wyk.log.INFO.20260426-114506.2765512
│   ├── llama_infer.keqing.wyk.log.INFO.20260426-115621.2766961
│   ├── llama_infer.keqing.wyk.log.WARNING.20260424-190124.2646288
│   ├── llama_infer.keqing.wyk.log.WARNING.20260424-190225.2646769
│   ├── llama_infer.keqing.wyk.log.WARNING.20260424-204851.2657945
│   ├── llama_infer.keqing.wyk.log.WARNING.20260424-210019.2659464
│   ├── llama_infer.keqing.wyk.log.WARNING.20260424-214021.2664058
│   ├── llama_infer.keqing.wyk.log.WARNING.20260424-214918.2665829
│   ├── llama_infer.keqing.wyk.log.WARNING.20260424-221316.2671236
│   ├── llama_infer.keqing.wyk.log.WARNING.20260424-221904.2672540
│   └── llama_infer.WARNING -> llama_infer.keqing.wyk.log.WARNING.20260424-221904.2672540
├── projectstructure.md
├── QWen
│   └── Qwen2.5-0.5B
│       ├── build
│       ├── config.json
│       ├── generation_config.json
│       ├── LICENSE
│       ├── merges.txt
│       ├── model.safetensors
│       ├── README.md
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.json
├── Qwen2.5-0.5B.bin
├── readme.md
├── test
│   ├── bench_matmul_vs_cublas.cpp
│   ├── CMakeLists.txt
│   ├── optimized
│   │   └── test_allocator.cpp
│   ├── test_cu
│   │   └── test_cu_wrap.cpp
│   ├── test_main.cpp
│   ├── test_model
│   │   └── test_llama_cpu.cpp
│   ├── test_op
│   │   ├── test_cu_add.cpp
│   │   ├── test_cu_emb.cpp
│   │   ├── test_cu_matmul.cpp
│   │   ├── test_cu_paged_mha.cpp
│   │   ├── test_cu_rmsnorm.cpp
│   │   ├── test_cu_rope.cpp
│   │   ├── test_cu_scale.cpp
│   │   ├── test_cu_softmax.cpp
│   │   ├── test_cu_swiglu.cpp
│   │   └── test_load.cpp
│   ├── test_tensor
│   │   ├── test_buffer.cpp
│   │   └── test_tensor.cpp
│   ├── utils.cu
│   └── utils.cuh
├── tmp
│   └── test.bin
└── tools
    ├── config.json
    ├── export_llama3.py
    ├── export_llama.py
    ├── export_qwen2.py
    ├── export_qwen3
    │   ├── config.py
    │   ├── load.py
    │   ├── model.py
    │   └── write_bin.py
    ├── model.py
    ├── model_qwen2.py
    └── __pycache__
        ├── model.cpython-310.pyc
        ├── model.cpython-313.pyc
        └── model_qwen2.cpython-310.pyc

69 directories, 460 files
