//
// Created by ubuntu on 7/6/24.
//

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

namespace leveldb {

class MyStats {
 public:
  std::atomic_uint64_t bytes_write;
  std::atomic_uint64_t bytes_flush_write;
  uint64_t compaction_time = 0;
  uint64_t cache_hit_count = 0;

  //  std::atomic_uint64_t write_sst_time;
  //  std::atomic_uint64_t write_vlog_time;
  //  std::atomic_uint64_t read_sst_time;
  //  std::atomic_uint64_t read_vlog_time;

  uint32_t original_value_size = 0;
  uint32_t var_key_value_size = 0;

  uint64_t max_vlog_size = 128 << 20;
  uint64_t clean_threshold = 666666;
  uint64_t clean_write_buffer_size = 1 << 20;
  uint32_t migrate_threshold = 666666;

  size_t max_num_log_item = 666666;

  size_t max_num_log = 1000;

  MyStats() : bytes_write(0), bytes_flush_write(0) {}

  void PrintStats(size_t done);

  void Reset() {
    bytes_write = 0;
    bytes_flush_write = 0;
    compaction_time = 0;
  }
};

extern MyStats my_stats;

}  // namespace leveldb
