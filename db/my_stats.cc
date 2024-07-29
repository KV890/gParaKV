//
// Created by ubuntu on 7/6/24.
//

#include "my_stats.h"

namespace leveldb {

MyStats my_stats;

void MyStats::PrintStats(size_t done_) {
  std::string directory = "/mnt/pmem";
  uintmax_t total_size = 0;

  try {
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(directory)) {
      if (entry.is_regular_file() && entry.path().extension() == ".ldb") {
        total_size += std::filesystem::file_size(entry.path());
      }
    }
  } catch (const std::filesystem::filesystem_error& e) {
  }

  directory = "/media/test";

  try {
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(directory)) {
      if (entry.is_regular_file()) {
        total_size += std::filesystem::file_size(entry.path());
      }
    }
  } catch (const std::filesystem::filesystem_error& e) {
  }

  fprintf(stdout,
          "flush到磁盘: %.0fMB,\t\t 写入磁盘: %.0fMB,\t\t 写放大: %.2f \n",
          (bytes_flush_write / 1048576.0), (bytes_write / 1048576.0),
          static_cast<double>(bytes_write) /
              static_cast<double>(bytes_flush_write));

  fprintf(
      stdout,
      "原始写入磁盘: %.0fMB,\t\t 新写入磁盘量： %.0fMB,\t\t 空间放大： %.2f \n",
      (static_cast<double>((original_value_size + 16) * done_) / 1048576.0),
      static_cast<double>(total_size) / 1048576.0,
      static_cast<double>(total_size) /
          static_cast<double>((original_value_size + 16) * done_));

  printf("compaction time: %lu\n", compaction_time);
  printf("cache hit count: %lu\n", cache_hit_count);

  //  printf("write sst time: %lu\n", write_sst_time.load());
  //  printf("write vlog time: %lu\n", write_vlog_time.load());
  //  printf("read sst time: %lu\n", read_sst_time.load());
  //  printf("read vlog time: %lu\n", read_vlog_time.load());
}

}  // namespace leveldb
