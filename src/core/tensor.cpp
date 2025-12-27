#include "tensor.h"
#include <numeric>
#include <iostream>
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape, Device device, Dtype dtype): shape_(shape), device_(device), dtype_(dtype) {

    if (!device_.is_cpu()) throw std::runtime_error("Only CPU device is supported in this version.");   // CUDA not implemented yet, working in arm64

    size_ = 1;

    for (int d : shape) size_ *= d; // Total size to allocate

    size_t bytes = size_ * dtype_size(dtype_);   // Calculate total bytes needed function in dtype.h
    buffer_ = std::make_unique<unsigned char[]>(bytes);  // Allocate raw memory 
    std::memset(buffer_.get(), 0, bytes); // Initialize to zero 

    compute_strides();  // Calculate strides automatically
}

int Tensor::ndim() const {
    return shape_.size();    // Number of dimensions
}

int Tensor::size() const {
    return size_;   // Total number of elements
}

const std::vector<int>& Tensor::shape() const {
    return shape_;    // Return shape vector
}

Dtype Tensor::dtype() const {
    return dtype_;  // Return data type
}

Device Tensor::device() const {
    return device_; // Return device
}

void* Tensor::raw_data() {
    return buffer_.get();   // Return raw data pointer
}

const void* Tensor::raw_data() const {
    return buffer_.get();       // Return const raw data pointer
}

template<typename T>
T* Tensor::data() {
    return reinterpret_cast<T*>(buffer_.get());   // Cast raw data to typed pointer
}
template<typename T>
const T* Tensor::data() const {
    return reinterpret_cast<const T*>(buffer_.get()); // Cast raw data to const typed pointer
}

void Tensor::fill_double(double value) {
    if (dtype_ == Dtype::Float32) {
        float* ptr = data<float>();
        for (int i = 0; i < size_; ++i) ptr[i] = static_cast<float>(value);
    } else if (dtype_ == Dtype::Float64) {
        double* ptr = data<double>();
        for (int i = 0; i < size_; ++i) ptr[i] = value;
    } else {
        throw std::runtime_error("fill not supported for this dtype");
    }
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    int stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

void Tensor::print() {
    std::cout << "Tensor shape: [ ";
    for (int s : shape_) std::cout << s << " ";
    std::cout << "]\n";
    for (float v : std::vector<float>(data<float>(), data<float>() + size_)) {
        std::cout << v << " ";
    }
    std::cout << "\n";
    std::cout << "Dtype: " << dtype_to_string(dtype_) << "\n";
    std::cout << "Device: " << (device_.is_cpu() ? "CPU" : "CUDA") << "\n";
    std::cout << "\n";
}

