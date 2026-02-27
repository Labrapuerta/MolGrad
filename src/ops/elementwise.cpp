#include "elementwise.h"
#include "shape/broadcast.h"
#include "backend/cpu/cpu_elementwise.h"
#include "backend/cuda/cuda_elementwise.h"
#include <iostream>


namespace ops {
    /// Sum of two tensors with broadcasting support

    Tensor add(const Tensor& a, const Tensor& b) {
        // broadcast the shape
        auto out_shape = shape::infer_broadcast_shape(a.shape(),b.shape());
        Tensor a_view = a.broadcast_to(out_shape);
        Tensor b_view = b.broadcast_to(out_shape);

        // Allocate memory for output
        Tensor out = Tensor(out_shape, a.dtype(), a.device());

        // Kernels will be separated
        if (a.device().is_cpu()) {
            cpu::elementwise_binary<float>(a_view, b_view, out, [](float x, float y) { return x + y; });
        } else {
            // For now, we will not implement CUDA kernel
            std::cerr << "CUDA kernel not implemented yet\n";
        }
        return out;

    }
    
    Tensor mul(const Tensor& a, const Tensor& b) {
        // broadcast the shape
        auto out_shape = shape::infer_broadcast_shape(a.shape(),b.shape());
        Tensor a_view = a.broadcast_to(out_shape);
        Tensor b_view = b.broadcast_to(out_shape);

        // Allocate memory for output
        Tensor out = Tensor(out_shape, a.dtype(), a.device());

        // Kernels will be separated
        if (a.device().is_cpu()) {
            cpu::elementwise_binary<float>(a_view, b_view, out, [](float x, float y) { return x * y; });
        } else {
            // For now, we will not implement CUDA kernel
            std::cerr << "CUDA kernel not implemented yet\n";
        }
        return out;
    }
    
    Tensor sub(const Tensor& a, const Tensor& b) {
        // broadcast the shape
        auto out_shape = shape::infer_broadcast_shape(a.shape(),b.shape());
        Tensor a_view = a.broadcast_to(out_shape);
        Tensor b_view = b.broadcast_to(out_shape);

        // Allocate memory for output
        Tensor out = Tensor(out_shape, a.dtype(), a.device());

        // Kernels will be separated
        if (a.device().is_cpu()) {
            cpu::elementwise_binary<float>(a_view, b_view, out, [](float x, float y) { return x - y; });
        } else {
            // For now, we will not implement CUDA kernel
            std::cerr << "CUDA kernel not implemented yet\n";
        }
        return out;
    }

    Tensor div(const Tensor& a, const Tensor& b) {
        // broadcast the shape
        auto out_shape = shape::infer_broadcast_shape(a.shape(),b.shape());
        Tensor a_view = a.broadcast_to(out_shape);
        Tensor b_view = b.broadcast_to(out_shape);

        // Allocate memory for output
        Tensor out = Tensor(out_shape, a.dtype(), a.device());

        // Kernels will be separated
        if (a.device().is_cpu()) {
            cpu::elementwise_binary<float>(a_view, b_view, out, [](float x, float y) { return x / y; });
        } else {
            // For now, we will not implement CUDA kernel
            std::cerr << "CUDA kernel not implemented yet\n";
        }
        return out;
    }

}