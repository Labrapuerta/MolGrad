#include "core/tensor.h"
#include <iostream>
#include <cassert>

int main() {
    Tensor x({4, 2});
    auto px = x.data<float>();

    for (int i = 0; i < 8; ++i)
        px[i] = i;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << px[i * 2 + j] << " ";
        }
        std::cout << "\n";
    }

    Tensor y = x.slice(0, 1, 3);

    std::cout << "x raw: " << x.raw_data() << "\n";
    std::cout << "y raw: " << y.raw_data() << "\n";


    std::cout << y.data<float>()[0] << "\n";  // should print x[1][0]

    y.data<float>()[0] = 99;
    std::cout << x.data<float>()[2] << "\n";  // 99

}