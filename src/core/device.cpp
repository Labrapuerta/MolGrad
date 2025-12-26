#include "device.h"

Device::Device(DeviceType type, int index)
    : type_(type), index_(index) {}

DeviceType Device::type() const {
    return type_;
}

int Device::index() const {
    return index_;
}

bool Device::is_cpu() const {
    return type_ == DeviceType::CPU;
}

bool Device::is_cuda() const {
    return type_ == DeviceType::CUDA;
}