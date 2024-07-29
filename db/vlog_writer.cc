// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "db/vlog_writer.h"

#include <stdint.h>

#include "leveldb/env.h"

#include "util/coding.h"
#include "util/crc32c.h"

namespace leveldb {
namespace log {

VWriter::VWriter(WritableFile* dest) : dest_(dest) {}

VWriter::~VWriter() {}

Status VWriter::AddRecord(const Slice& slice) {
  Status s = dest_->Append(slice);
  if (s.ok()) {
    s = dest_->Flush();
  }
  return s;
}

}  // namespace log
}  // namespace leveldb
