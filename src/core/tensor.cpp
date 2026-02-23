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

Tensor Tensor::broadcast_to(const std::vector<int>& target_shape) const {
    std::vector<int> infered_shape = shape::infer_broadcast_shape(shape_, target_shape);
    // Maybe redundant check, inside of broadcast there's already a check for broadcast compatibility, but we can keep it for safety
    if (infered_shape != target_shape) throw std::invalid_argument("Inferred broadcast shape does not match target shape");

    int old_rank = shape_.size();
    int new_rank = infered_shape.size();
    std::vector<int> new_strides(new_rank);

    // A lot of comments cause I'm lost in here
    for (int i = 0; i < new_rank; i++) {
        // Corresponding dimension in the original shape (if it exists)
        int new_dim = infered_shape[new_rank - 1 - i];
        
        int old_dim = (old_rank - 1 - i >= 0) ? shape_[old_rank - 1 - i] : 1; // If old rank is smaller, treat missing dims as 1

        int old_stride = (old_rank - 1 - i >= 0) ? strides_[old_rank - 1 - i] : 0; // If old rank is smaller, missing dims have stride 0

        if (new_dim == old_dim) {
            new_strides[new_rank - 1 - i] = old_stride; // Same, keep
        } else if (old_dim == 1) {
            new_strides[new_rank - 1 - i] = 0; // Broadcasted dimension, stride becomes 0
        } else {
            throw std::invalid_argument("Shapes cannot be broadcasted");
        }
    }

    return Tensor(storage_, offset_, infered_shape, new_strides, dtype_); // New tensor with broadcasted shape and strides
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


///  Apparently, contiguous and clone have very similar implementations 

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;  // Already contiguous
    }
    return clone();  // Use clone to create a contiguous copy
}

Tensor Tensor::clone() const {   // clone will always crate a new storage
    Tensor out(shape_, dtype_, device());   // New tensor with same shape and dtype
    if (is_contiguous()) {
        size_t bytes = numel() * dtype_size(dtype_);
        std::memcpy(out.raw_data(), this->raw_data(), bytes);  // Direct copy if both are contiguous
        return out;
    }
    // Non-contiguous copy
    std::vector<int> idx(shape_.size(), 0);
    size_t total_elements = numel();
    for (size_t i = 0; i < total_elements; ++i) {
        out.at<float>(idx) = this->at<float>(idx);
        // Increment index
        for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
            idx[d]++;
            if (idx[d] < shape_[d]) break;
            idx[d] = 0;
        }
    }
    return out;
} 

///

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

bool Tensor::is_contiguous() const {
    size_t expected_stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (strides_[i] != expected_stride) return false;
        expected_stride *= shape_[i];
    }
    return true;
}

void* Tensor::raw_data() {
    return static_cast<char*>(storage_ -> data()) + offset_ ;   // Return raw data pointer
}

const void* Tensor::raw_data() const {
    return static_cast<const char*>(storage_ -> data()) + offset_ ;       // Return const raw data pointer
}

size_t Tensor::numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>()); // Total number of elements
}



