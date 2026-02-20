#include "broadcast.h"

namespace shape { 
    std::vector<int> infer_broadcast_shape(const std::vector<int>& shape1, const std::vector<int>& shape2) {
        const int rank_1 = shape1.size();
        const int rank_2 = shape2.size();
        const int out_rank = std::max(rank_1, rank_2);

        std::vector<int> result(out_rank, 1);

        for (size_t i = 0; i < out_rank; ++i) {
            int dim1 = (i < out_rank - rank_1) ? 1 : shape1[i - (out_rank - rank_1)];
            int dim2 = (i < out_rank - rank_2) ? 1 : shape2[i - (out_rank - rank_2)];

            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw std::invalid_argument("Shapes cannot be broadcasted");
            }
            result[i] = std::max(dim1, dim2);
        }
        return result;
    }

}