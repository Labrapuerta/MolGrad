#include "tensor.h"
#include <numeric>
#include <iostream>
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape, Device device, Dtype dtype): shape_(shape), device_(device), dtype_(dtype) {

    if (!device_.is_cpu()) {
        throw std::runtime_error("Only CPU device is supported in this version.");   // CUDA not implemented yet, working in arm64
    }

    size_ = 1;

    for (int d : shape) size_ *= d; // Total size to allocate

    data_ = std::make_unique<float[]>(size_);    // Allocate memory

    std::fill(data_.get(), data_.get() + size_, 0.0f);  // Initialize to zero

    compute_strides();  // Calculate strides automatically
}

float* Tensor::data() {
    return data_.get();
}

const float* Tensor::data() const {
    return data_.get();
}

int Tensor::ndim() const {
    return shape_.size(); 
}

int Tensor::size() const {
    return size_;
}

const std::vector<int>& Tensor::shape() const {
    return shape_;
}

void Tensor::fill(float value) {
    for (int i = 0; i < size_; ++i)
        data_[i] = value;
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
    std::cout << "]\nData: ";
    for (int i = 0; i < size_; ++i) {
        std::cout << data_.get()[i] << " ";
    }
    std::cout << "\nDevice: " << (device_.is_cpu() ? "CPU" : "CUDA") << "\n";
    std::cout << "\n";
}

