#include "tensor.h"
#include <numeric>
#include <iostream>
#include <stdexcept>

// Constructor that allocates new storage
Tensor::Tensor(const std::vector<int>& shape, Dtype dtype, Device device): shape_(shape), dtype_(dtype), offset_(0) {

    if (shape_.empty()) throw std::runtime_error("Tensor shape cannot be empty");   // Validate shape

    size_t elements = 1;
    for (int d : shape_) elements *= d; // Total size to allocateb

    size_t bytes = elements * dtype_size(dtype_);   // Calculate total bytes needed
    storage_ = std::make_shared<Storage>(bytes, device);  // Allocate storage
    compute_contiguous_strides();  // Calculate strides automatically
}

// Constructor from existing storage
Tensor::Tensor(std::shared_ptr<Storage> storage, size_t offset, 
               const std::vector<int>& shape,
               const std::vector<int>& strides,
               Dtype dtype)
    : storage_(storage), offset_(offset), shape_(shape), strides_(strides), dtype_(dtype) {

        if (!storage_) throw std::runtime_error("Storage pointer cannot be null");

        if (shape_.size() != strides_.size()) throw std::runtime_error("Shape and strides must have the same number of dimensions");
}

void Tensor::compute_contiguous_strides() {
    strides_.resize(shape_.size());
    int stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i){
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

Tensor Tensor::slice(int dim, int start, int end) const {
    // Validate inputs
    if (dim < 0 || dim >= shape_.size()) throw std::runtime_error("slice: invalid dimension");

    if (start < 0 || end > shape_[dim] || start >= end) throw std::runtime_error("slice: invalid range");

    std::vector<int> new_shape = shape_;
    new_shape[dim] = end - start;

    // Compute new offset
    size_t element_offset = start * strides_[dim];
    size_t byte_offset = element_offset * dtype_size(dtype_);
    size_t new_offset = offset_ + byte_offset;

    return Tensor(storage_, new_offset, new_shape, strides_, dtype_);
}

float& Tensor::operator()(const std::vector<int>& indices) {
    size_t offset = compute_offset(indices);
    return *(reinterpret_cast<float*>(static_cast<char*>(storage_->data()) + offset));
}

size_t Tensor::compute_offset(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) throw std::runtime_error("compute_offset: indices size must match number of dimensions");
    size_t offset = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) throw std::runtime_error("compute_offset: index out of bounds");
        offset += indices[i] * strides_[i] * dtype_size(dtype_);
    }
    return offset;
}

void* Tensor::raw_data() {
    return static_cast<char*>(storage_ -> data()) + offset_ ;   // Return raw data pointer
}

const void* Tensor::raw_data() const {
    return static_cast<const char*>(storage_ -> data()) + offset_ ;       // Return const raw data pointer
}

template<typename T>
T* Tensor::data() {
    return reinterpret_cast<T*>(raw_data()); // Return typed data pointer
}

template<typename T>
const T* Tensor::data() const {
    return reinterpret_cast<const T*>(raw_data()); // Return const typed data pointer
}

// Explicit template instantiations
template float* Tensor::data<float>();
template float const* Tensor::data<float>() const;

size_t Tensor::numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>()); // Total number of elements
}



