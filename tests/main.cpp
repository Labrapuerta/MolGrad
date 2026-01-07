#include "core/tensor.h"
#include <iostream>
#include <cassert>

int main() {
    Tensor x({4, 2});
    auto px = x.data<float>();

    for (int i = 0; i < 8; ++i)
        px[i] = i;

    std::cout<< "Offset of x: " << x({3,0}) << "\n";
    std::cout << "Access via at(2,1): " << x.at<float>({2,1}) << "\n";

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
    std::cout << "Numel x:" << x.numel() << "\n"; // should print 4   
    std::cout << "Numel y:" << y.numel() << "\n"; // should print 2

    std::cout << "Dims x:" << x.n_dim() << "\n"; // should print 2  
    std::cout << "Dims y:" << y.n_dim() << "\n"; //

    y.data<float>()[0] = 99;
    std::cout << x.data<float>()[2] << "\n";  // 99

}