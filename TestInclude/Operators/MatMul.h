#include <xtensor.hpp>
#include <Operators/Gemm.h> 


template <typename Ta, typename Tb, typename Ty>
void MatMul(const xt::xarray<Ta> a, const xt::xarray<Tb> b, xt::xarray<Ty>& y)
{
    if ((a.shape().size() == 1) && (b.shape().size() == 1))
    {

        y = xt::linalg::vdot(a, b);

    }
    else if ((a.shape().size() <= 2) && (b.shape().size() <= 2)) {
        y = xt::linalg::dot(a, b);;
    }
    else
    {
        std::vector<size_t> batchShape(a.shape().begin(), a.shape().end() - 2);
        std::vector<size_t> resultShape = batchShape;
        resultShape.push_back(a.shape()[a.shape().size() - 2]);
        resultShape.push_back(b.shape()[b.shape().size() - 1]);
        y = xt::empty<Ty>(resultShape);
        
        // calculate Batch size
        size_t batchSize = 1;
        for (auto dim : batchShape) batchSize *= dim;
        for (size_t i = 0; i < batchSize; i++) {
            // Calculate indexes for bachtes
            std::vector<size_t> idx(batchShape.size());
            size_t tmp = i;
            for (int j = batchShape.size() - 1; j >= 0; j--) {
                idx[j] = tmp % batchShape[j];
                tmp /= batchShape[j];
            }
            //  go on the right level for batch
            auto aView = xt::eval(xt::view(a, xt::all()));
            auto bView = xt::eval(xt::view(b, xt::all()));
            for (size_t j = 0; j < batchShape.size(); j++) {
                aView = xt::view(aView, idx[j], xt::all());
                bView = xt::view(bView, idx[j], xt::all());
            }
            // calculate dot for batche
            auto res = xt::linalg::dot(aView, bView);
            // Genrate view for result
            xt::xdynamic_slice_vector  sv;
            for (size_t j = 0; j < batchShape.size(); j++) {
                sv.push_back(xt::keep(idx[j]));

            }
            auto outView = xt::dynamic_view(y, sv);
            outView = res;
        }
    }
}
