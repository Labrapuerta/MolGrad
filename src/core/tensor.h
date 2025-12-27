#pragma once
#include <vector>
#include <memory>
#include "device.h"
#include "dtype.h"

class Tensor {
public: 
    Tensor(const std::vector<int>& shape, Device device = Device(), Dtype dtype = Dtype::Float32);

    int ndim() const;  // Number of dimensions
    int size() const;  // Total number of elements
    const std::vector<int>& shape() const;

    Dtype dtype() const;
    Device device() const;

    void* raw_data();
    const void* raw_data() const;

    template<typename T>
    T* data();

    template<typename T>
    const T* data() const;

    void fill_double(double value);    // Fill tensor with a specific value
    void print();               // Print tensor details

private:
    void compute_strides();

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    int size_; // Total number of elements

    Device device_;
    Dtype dtype_;

    std::unique_ptr<unsigned char[]> buffer_;
};