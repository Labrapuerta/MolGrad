#pragma once
#include <vector>
#include <memory>
#include "core/device.h"
#include "core/dtype.h"
#include "core/storage.h"
#include "shape/broadcast.h"


class Tensor {
public: 
    // Root constructor, allocates new storage
    Tensor(const std::vector<int>& shape, Dtype dtype = Dtype::Float32, Device device = Device());  // basic constructor

    // Constructor from existing storage (no allocation)
    Tensor(std::shared_ptr<Storage> storage, size_t offset, 
            const std::vector<int>& shape,
            const std::vector<int>& strides,
            Dtype dtype); 

    Tensor broadcast_to(const std::vector<int>& target_shape) const; // Return a new tensor broadcasted to target shape
    
    Tensor slice(int dim, int start, int end) const;  // Slice tensor along a dimension

    Tensor contiguous() const; // Return a contiguous copy of the tensor

    bool is_contiguous() const; // Check if tensor is contiguous in memory

    Tensor clone() const; // Return a deep copy of the tensor

    float & operator()(const std::vector<int>& indices); // Returns value at given index, always as a float  **
    
    // Typed accessors
    template<typename T>
    T& at(const std::vector<int>& indices) {
        if (dtype_ != dtype_of<T>())
            throw std::runtime_error("dtype mismatch");

        size_t offset = compute_offset(indices);
        return *reinterpret_cast<T*>(static_cast<unsigned char*>(storage_->data()) + offset);
    }

    template<typename T>
    const T& at(const std::vector<int>& indices) const {
        if (dtype_ != dtype_of<T>())
            throw std::runtime_error("dtype mismatch");

        size_t offset = compute_offset(indices);
        return *reinterpret_cast<const T*>(static_cast<unsigned char*>(storage_->data()) + offset);
    }

    // Data 

    template<typename T>
    T* data() {
        if (dtype_ != dtype_of<T>())
            throw std::runtime_error("dtype mismatch");

        return reinterpret_cast<T*>(static_cast<unsigned char*>(storage_->data()) + offset_);
    }

    template<typename T>
    const T* data() const {
        if (dtype_ != dtype_of<T>())
            throw std::runtime_error("dtype mismatch");

        return reinterpret_cast<const T*>(static_cast<unsigned char*>(storage_->data()) + offset_);
    }

    void* raw_data();
    const void* raw_data() const;

    // Metadata 

    

    const std::vector<int>& shape() const {return shape_;}    // Return shape vector
    const std::vector<int>& strides() const {return strides_;} // Return strides vector
    Dtype dtype() const {return dtype_;}  // Return data type
    Device device() const {return storage_ -> device();} // Return device from storage
    std::shared_ptr<Storage> storage() const {return storage_;} // Return underlying storage
    size_t numel() const; // Return number of elements
    size_t n_dim() const {return shape_.size();} // Return number of dimensions
    
private:
    void compute_contiguous_strides(); // Compute strides for contiguous layout
    size_t compute_offset(const std::vector<int>& indices) const; // Compute offset in bytes for given indices


private:
    std::shared_ptr<Storage> storage_; // Underlying storage
    size_t offset_; // Offset in storage


    std::vector<int> shape_; 
    std::vector<int> strides_;

    Dtype dtype_;


};