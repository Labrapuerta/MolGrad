#include "core/storage.h"
#include <cstdlib>
#include <stdexcept>

Storage::Storage(size_t size_bytes, Device device)
    : size_bytes_(size_bytes), device_(device) {

    if (device_.is_cpu()) {
        void* ptr = std::malloc(size_bytes_);
        if (!ptr) throw std::bad_alloc();
        data_ = std::shared_ptr<void>(ptr, std::free);  // Custom deleter to free memory

    } else throw std::runtime_error("Cuda not implemented yet");


}

void* Storage::data() {
    return data_.get();   // Return raw data pointer
}

const void* Storage::data() const {
    return data_.get();       // Return const raw data pointer
}

size_t Storage::size_bytes() const {
    return size_bytes_;   // Return size in bytes
}

Device Storage::device() const {
    return device_; // Return device
}

