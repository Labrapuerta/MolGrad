#include "core/tensor.h"
#include <iostream>
#include <cassert>

int main() {
    Tensor t({7,2, 3}, DeviceType::CPU, Dtype::Float32);
    t.fill_double(1.5);
    float* p = t.data<float>();
    for (int i = 0; i < t.size(); ++i)
        assert(p[i] == 1.5f);

    Tensor b({4}, DeviceType::CPU, Dtype::Float64);
    b.fill_double(2.0);
    double* q = b.data<double>();
    assert(q[0] == 2.0);

    std::cout << "Tensor tests passed\n";
    return 0;
}