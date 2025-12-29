#pragma once

#include <memory>
#include <cstddef>
#include "core/device.h"

class Storage {
public:
    Storage(size_t size_bytes, Device device);

    void* data();
    const void* data() const;

    size_t size_bytes() const;
    Device device() const;

private:
    std::shared_ptr<void> data_;  // Shared pointer to manage memory
    size_t size_bytes_;
    Device device_;
};

