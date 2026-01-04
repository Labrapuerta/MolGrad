#pragma once
#include <vector>
#include <memory>
#include "core/device.h"
#include "core/dtype.h"
#include "core/storage.h"


class Tensor {
public: 
    // Root constructor, allocates new storage
    Tensor(const std::vector<int>& shape, Dtype dtype = Dtype::Float32, Device device = Device());  // basic constructor

    // Constructor from existing storage (no allocation)
    Tensor(std::shared_ptr<Storage> storage, size_t offset, 
            const std::vector<int>& shape,
            const std::vector<int>& strides,
            Dtype dtype); 
    
    Tensor slice(int dim, int start, int end) const;
    
    void* raw_data();
    const void* raw_data() const;

    template<typename T>
    T* data();

    template<typename T>
    const T* data() const;

    // Metadata 

    const std::vector<int>& shape() const {return shape_;}    // Return shape vector
    const std::vector<int>& strides() const {return strides_;} // Return strides vector
    Dtype dtype() const {return dtype_;}  // Return data type
    Device device() const {return storage_ -> device();} // Return device from storage
    std::shared_ptr<Storage> storage() const {return storage_;} // Return underlying storage

    size_t numel() const;

private:
    void compute_contiguous_strides();

private:
    std::shared_ptr<Storage> storage_; // Underlying storage
    size_t offset_; // Offset in storage


    std::vector<int> shape_; 
    std::vector<int> strides_;

    Dtype dtype_;


};