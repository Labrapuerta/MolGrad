#include "core/tensor.h"
#include "shape/broadcast.h"
#include <iostream>
#include <cassert>

void print_shape(const Tensor& t) {
    for (int d : t.shape()) std::cout << d << " ";
    std::cout << "\n";
}

int main() {

    std::cout << "\n=== BASIC CREATION ===\n";

    Tensor x({4, 2});
    auto px = x.data<float>();

    for (int i = 0; i < 8; ++i)
        px[i] = static_cast<float>(i);

    assert(x.numel() == 8);
    assert(x.n_dim() == 2);
    assert(x.at<float>({2,1}) == 5);
    assert(x.at<float>({3,0}) == 6);

    std::cout << "Basic indexing OK\n";


    std::cout << "\n=== SLICE TEST ===\n";

    Tensor y = x.slice(1, 1, 2);  // slice dim=1, start=1, size=2

    std::cout << "x raw: " << x.raw_data() << "\n";
    std::cout << "y raw: " << y.raw_data() << "\n";

    // y should share storage
    y.data<float>()[0] = 99.0f;

    // x must reflect change
    assert(x.data<float>()[1] == 99.0f);

    std::cout << "Slice shares storage OK\n";


    std::cout << "\n=== CLONE TEST ===\n";

    Tensor z = y.clone();

    std::cout << "y raw: " << y.raw_data() << "\n";
    std::cout << "z raw: " << z.raw_data() << "\n";

    // Must NOT share storage
    assert(z.raw_data() != y.raw_data());

    z.data<float>()[0] = -1.0f;

    // Changing clone must NOT affect original
    assert(y.data<float>()[0] == 99.0f);

    std::cout << "Clone deep copy OK\n";


    std::cout << "\n=== CONTIGUOUS TEST ===\n";

    Tensor y_contig = y.contiguous();

    assert(y_contig.raw_data() != y.raw_data());
    assert(y_contig.is_contiguous());

    std::cout << "Contiguous copy OK\n";


    std::cout << "\n=== BROADCAST SHAPE INFERENCE TEST ===\n";

    auto shape1 = shape::infer_broadcast_shape({3,1}, {1,4});
    assert(shape1 == std::vector<int>({3,4}));

    auto shape2 = shape::infer_broadcast_shape({5,1,7}, {1,7});
    assert(shape2 == std::vector<int>({5,1,7}));

    std::cout << "Broadcast shape inference OK\n";


    std::cout << "\n=== BROADCAST VIEW TEST ===\n";

    Tensor a({3,1});
    auto pa = a.data<float>();
    pa[0] = 10;
    pa[1] = 20;
    pa[2] = 30;

    Tensor b = a.broadcast_to({3,4});

    print_shape(b);

    // stride for broadcasted dimension should be 0
    // check that all columns read same value

    assert(b.at<float>({0,0}) == 10);
    assert(b.at<float>({0,3}) == 10);

    assert(b.at<float>({2,0}) == 30);
    assert(b.at<float>({2,3}) == 30);

    std::cout << "Broadcast stride=0 behavior OK\n";


    std::cout << "\n=== BROADCAST + CLONE MATERIALIZATION ===\n";

    Tensor c = b.clone();

    // c must be contiguous
    assert(c.is_contiguous());

    // changing c should NOT affect a
    c.data<float>()[0] = -100;

    assert(a.data<float>()[0] == 10);

    std::cout << "Broadcast clone materialization OK\n";


    std::cout << "\nALL TESTS PASSED \n";
}
