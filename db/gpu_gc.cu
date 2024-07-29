//
// Created by ubuntu on 7/22/24.
//

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include "gpu_gc.h"

void GPUGC::MallocMemory() {
  cudaMalloc(&gpu_flags, max_num_log * max_num_log_item);
  cudaMemset(gpu_flags, 1, max_num_log * max_num_log_item);

  cudaMalloc(&invalid_count, max_num_log * sizeof(uint32_t));
  cudaMemset(invalid_count, 0, max_num_log * sizeof(uint32_t));

  cudaStreamCreate(&stream);
}

GPUGC::~GPUGC() {
  cudaFree(gpu_flags);
  cudaFree(invalid_count);

  cudaStreamDestroy(stream);
}

__global__ void MarkKernel(GPUKeyValue* invalid_pos_d, size_t invalid_size,
                           uint8_t* gpu_flags, uint32_t max_num_log_item,
                           uint32_t var_key_value_size, uint32_t* invalid_count,
                           uint32_t max_size) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= invalid_size) return;

  uint32_t vlog_num = GPUDecodeFixed32(invalid_pos_d[tid].value);
  uint32_t invalid_pos = GPUDecodeFixed32(invalid_pos_d[tid].value + 4);

  uint32_t idx = (vlog_num - 1) * max_num_log_item +
                 (invalid_pos - 12) / var_key_value_size;

  if (invalid_pos < 12 || idx >= max_size) return;

  gpu_flags[idx] = 0;

  atomicAdd(&invalid_count[vlog_num - 1], 1);
}

void GPUGC::Mark(GPUKeyValue* invalid_pos_d, size_t invalid_size) {
  // 计算块和线程数
  size_t threadsPerBlock = 256;
  size_t blocksPerGrid = (invalid_size + threadsPerBlock - 1) / threadsPerBlock;

  uint32_t max_size = max_num_log_item * max_num_log;

  MarkKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      invalid_pos_d, invalid_size, gpu_flags, max_num_log_item,
      leveldb::my_stats.var_key_value_size + 12, invalid_count, max_size);

  CHECK(cudaStreamSynchronize(stream));
}

/*
__global__ void TriggerGCKernel(const uint32_t* invalid_count,
                                uint32_t* vlog_num_d, uint32_t max_num_log,
                                uint32_t clean_threshold, uint32_t* count_d) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= max_num_log) return;

  if (invalid_count[tid] >= clean_threshold) {
    *vlog_num_d = tid;
    *count_d = invalid_count[tid];
    return;
  }
}

bool GPUGC::TriggerGC() {
  uint32_t* vlog_num_d;
  cudaMalloc(&vlog_num_d, sizeof(uint32_t));
  cudaMemset(vlog_num_d, 0, sizeof(uint32_t));

  uint32_t* count_d;
  cudaMalloc(&count_d, sizeof(uint32_t));

  // 计算块和线程数
  size_t threadsPerBlock = 256;
  size_t blocksPerGrid = (max_num_log + threadsPerBlock - 1) / threadsPerBlock;

  TriggerGCKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      invalid_count, vlog_num_d, max_num_log, leveldb::my_stats.clean_threshold,
      count_d);

  CHECK(cudaStreamSynchronize(stream));

  uint32_t vlog_num;
  cudaMemcpy(&vlog_num, vlog_num_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  if (vlog_num != 0) {
    uint32_t count;
    cudaMemcpy(&count, count_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    triggered_vlog_num = vlog_num;
    triggered_invalid_count = count;
  }

  cudaFree(vlog_num_d);
  cudaFree(count_d);

  return vlog_num != 0;
}
*/

// 定义谓词结构
struct is_count_greater_than_threshold {
  const uint32_t threshold;
  is_count_greater_than_threshold(uint32_t threshold) : threshold(threshold) {}

  __host__ __device__ bool operator()(const uint32_t count) const {
    return count >= threshold;
  }
};

bool GPUGC::TriggerGC() {
  thrust::device_vector<uint32_t> d_invalid_count(invalid_count,
                                                  invalid_count + max_num_log);
  thrust::device_vector<uint32_t> d_indices(max_num_log);

  auto counting_iter = thrust::counting_iterator<uint32_t>(0);

  // 复制符合条件的索引到 d_indices
  auto end_iter = thrust::copy_if(
      counting_iter, counting_iter + max_num_log, d_invalid_count.begin(),
      d_indices.begin(),
      is_count_greater_than_threshold(leveldb::my_stats.clean_threshold));

  // 计算符合条件的元素数量
  size_t num_filtered = end_iter - d_indices.begin();

  if (num_filtered > 0) {
    // 获取第一个符合条件的索引和计数值
    uint32_t vlog_num;
    cudaMemcpy(&vlog_num, thrust::raw_pointer_cast(&d_indices[0]),
               sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t count;
    cudaMemcpy(&count, thrust::raw_pointer_cast(&d_invalid_count[vlog_num]),
               sizeof(uint32_t), cudaMemcpyDeviceToHost);

    triggered_vlog_num = vlog_num + 1;
    triggered_invalid_count = count;

    return true;
  }

  return false;
}

__global__ void GPUGCKernel(char* vlog_d, char* output_d, uint8_t* flags,
                            uint32_t max_num_log_item,
                            uint32_t triggered_vlog_num, uint32_t* global_count,
                            uint32_t var_key_value_size) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= max_num_log_item) return;

  flags = flags + (triggered_vlog_num - 1) * max_num_log_item;

  if (flags[tid] == 0) return;

  uint32_t index = atomicAdd(global_count, 1);
  memcpy(output_d + index * var_key_value_size,
         vlog_d + tid * var_key_value_size, var_key_value_size);
}

void GPUGC::BeginGPUGC(const char* vlog, size_t vlog_size, char** output,
                       size_t& output_size) {
  char* vlog_d;
  cudaMalloc(&vlog_d, vlog_size);
  cudaMemcpy(vlog_d, vlog, vlog_size, cudaMemcpyHostToDevice);

  uint32_t* global_count;
  cudaMalloc(&global_count, max_num_log_item * sizeof(uint32_t));
  cudaMemset(global_count, 0, max_num_log_item * sizeof(uint32_t));

  output_size = vlog_size - triggered_invalid_count *
                                (leveldb::my_stats.var_key_value_size + 12);
  char* output_d;
  cudaMalloc(&output_d, output_size);

  // 计算块和线程数
  size_t threadsPerBlock = 1024;
  size_t blocksPerGrid =
      (max_num_log_item + threadsPerBlock - 1) / threadsPerBlock;

  GPUGCKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      vlog_d, output_d, gpu_flags, max_num_log_item, triggered_vlog_num,
      global_count, leveldb::my_stats.var_key_value_size + 12);

  CHECK(cudaStreamSynchronize(stream));

  *output = new char[output_size];
  cudaMemcpy(*output, output_d, output_size, cudaMemcpyDeviceToHost);

  cudaFree(output_d);
  cudaFree(global_count);
  cudaFree(vlog_d);
}

void GPUGC::CleanGC() {
  cudaMemset(invalid_count + triggered_vlog_num - 1, 0, sizeof(uint32_t));

  triggered_vlog_num = 0;
  triggered_invalid_count = 0;
}
