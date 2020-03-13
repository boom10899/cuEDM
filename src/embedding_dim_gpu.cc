#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <arrayfire.h>

#include "embedding_dim_gpu.h"
#include "stats.h"

EmbeddingDimGPU::EmbeddingDimGPU(uint32_t max_E, uint32_t tau, uint32_t Tp,
                                 bool verbose)
    : EmbeddingDim(max_E, tau, Tp, verbose),
      knn(new NearestNeighborsGPU(tau, Tp, verbose)),
      simplex(new SimplexCPU(tau, Tp, verbose)), rhos(max_E)
{
    n_devs = af::getDeviceCount();

    luts.resize(n_devs);
    buffers.resize(n_devs);
}

// clang-format off
uint32_t EmbeddingDimGPU::run(const Series &ts, double &timer_knn_elapsed,
                              double &timer_lookup_elapsed)
{
    #pragma omp parallel num_threads(n_devs)
    {
        Timer timer_knn;
        Timer timer_lookup;

        #ifdef _OPENMP
        uint32_t dev_id = omp_get_thread_num();
        #else
        uint32_t dev_id = 0;
        #endif

        af::setDevice(dev_id);

        // Split input into two halves
        const auto library = ts.slice(0, ts.size() / 2);
        const auto target = ts.slice(ts.size() / 2);

        #pragma omp for schedule(dynamic)
        for (auto E = 1u; E <= max_E; E++) {
            timer_knn.start();
            knn->compute_lut(luts[dev_id], library, target, E, E + 1);
            luts[dev_id].normalize();
            timer_knn.stop();

            timer_lookup.start();
            const auto prediction =
                simplex->predict(buffers[dev_id], luts[dev_id], library, E);
            const auto shifted_target = simplex->shift_target(target, E);

            rhos[E - 1] = corrcoef(prediction, shifted_target);
            timer_lookup.stop();
        }

        #pragma omp critical 
        {
            timer_knn_elapsed += timer_knn.elapsed();
            timer_lookup_elapsed += timer_lookup.elapsed();
        }
    }

    const auto it = std::max_element(rhos.begin(), rhos.end());
    const auto best_E = it - rhos.begin() + 1;

    timer_knn_elapsed = timer_knn_elapsed / n_devs;
    timer_lookup_elapsed = timer_lookup_elapsed / n_devs;

    return best_E;
}
// clang-format on
