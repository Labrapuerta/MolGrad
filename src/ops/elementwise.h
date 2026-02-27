#pragma once
#include "core/tensor.h"

namespace ops {
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor mul(const Tensor& a, const Tensor& b);
    Tensor sub(const Tensor& a, const Tensor& b);
    Tensor div(const Tensor& a, const Tensor& b);
}