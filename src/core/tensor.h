#pragma once
#include <vector>
#include <memory>
#include "device.h"
#include "dtype.h"

class Tensor {
public: 
    Tensor(const std::vector<int>& shape, Device device = Device(), Dtype dtype = Dtype::Float32);

    float* data();
    const float* data() const;

    int ndim() const;  // Number of dimensions
    int size() const;
    const std::vector<int>& shape() const;

    void fill(float value);    // Fill tensor with a specific value
    void print();

private:
    void compute_strides();

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    int size_; // Total number of elements

    Device device_;
    Dtype dtype_;

    std::unique_ptr<float[]> data_;

};