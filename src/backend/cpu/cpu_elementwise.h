#pragma once
#include "core/tensor.h"
#include <iostream>

namespace cpu {
    // Binary elementwise addition of two tensors with broadcasting support
    template<typename T, typename Func>
    void elementwise_binary(const Tensor& a, const Tensor& b, Tensor& out, Func func) {
        size_t total_elements = out.numel();
        std::cout << "Total elements to compute: " << total_elements << "\n";
        /// Try contiguous fast path first
        if (a.is_contiguous() && b.is_contiguous() && out.is_contiguous()) {
            std::cout << "All tensors are contiguous, using simple pointer arithmetic\n";
            T* out_ptr = out.data<T>();
            const T* a_ptr = a.data<T>();
            const T* b_ptr = b.data<T>();
            for (size_t i = 0; i < total_elements; ++i) {
                out_ptr[i] = func(a_ptr[i], b_ptr[i]); /// Sum a and b at the current index and store in out
            }
            return;
        }
        // For non-contiguous tensors, we need to compute the correct offsets
        std::cout << "Non-contiguous tensors detected, using index-based access\n";
        std::vector<int> idx(out.shape().size(), 0);
            for (size_t i = 0; i < total_elements; ++i) {
                /// This is very inefficient, but it is just a reference implementation to verify correctness. We will optimize this later.
                out.at<T>(idx) = func(a.at<T>(idx), b.at<T>(idx)); /// Sum a and b at the current index and store in out
                // Increment index  
                for (int d = static_cast<int>(out.n_dim()) - 1; d >= 0; --d) {
                    idx[d]++;
                    if (idx[d] < out.shape()[d]) break;
                    idx[d] = 0;
                }
        }
    } 
}

