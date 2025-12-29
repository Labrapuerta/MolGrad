#include "core/tensor.h"
#include <iostream>
#include <cassert>

int main() {
    Tensor x({4, 2});
    auto p = x.data<float>();
    p[0] = 1;

    Tensor y(x.storage(), 4 * sizeof(float), {2}, {1}, Dtype::Float32);

    std::cout << "Memory address x: " << x.raw_data() << std::endl;
    std::cout << "Memory address y: " << y.raw_data() << std::endl;


    y.data<float>()[0] = 99;

    // x = [1, ?, 99, ?]
}