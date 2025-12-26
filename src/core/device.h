#pragma once

enum class DeviceType {
    CPU,
    CUDA
};

class Device {
public:
    Device(DeviceType type = DeviceType::CPU, int index = -1);  // Default to CPU device
    DeviceType type() const;   // prepared for future use
    int index() const;   // prepared for future use

    bool is_cpu() const;
    bool is_cuda() const;  

private:
    DeviceType type_;
    int index_;
};