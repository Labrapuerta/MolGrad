#pragma once
#include <vector>
#include <memory>
#include "device.h"

class Tensor {
public: 
    Tensor(const std::vector<int>& shape, Device device = Device());


    float* data();
    const float* data() const;

    int ndim() const;
    int size() const;
    const std::vector<int>& shape() const;

    void fill(float value);    
    void print();

private:
    void compute_strides();

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    int size_; // Total number of elements

    Device device_;

    std::unique_ptr<float[]> data_;

};