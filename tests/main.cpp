#include "core/tensor.h"
#include "core/device.h"
#include <iostream>

int main() {
    Tensor t({2, 3});
    t.print();
    std::cout << "Filling tensor with 5.0\n";
    t.fill(5.0f);
    t.print();
    std::cout << "Tensor shape: ";
    return 0;
}