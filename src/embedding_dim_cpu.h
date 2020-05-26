#ifndef __EMBEDDING_DIM_CPU_H__
#define __EMBEDDING_DIM_CPU_H__

#include <memory>

#include "embedding_dim.h"
#include "lut.h"
#include "nearest_neighbors_cpu.h"
#include "simplex_cpu.h"
#include "timer.h"

class EmbeddingDimCPU : public EmbeddingDim
{
public:
    EmbeddingDimCPU(uint32_t max_E, uint32_t tau, uint32_t Tp, bool verbose)
        : EmbeddingDim(max_E, tau, Tp, verbose),
          knn(new NearestNeighborsCPU(tau, Tp, verbose)),
          simplex(new SimplexCPU(tau, Tp, verbose)), rhos(max_E)
    {
    }

    uint32_t run(const Series &ts) override;

    double get_timer_knn_sum() override;
    double get_timer_lookup_sum() override;
    double get_timer_cpu_to_gpu_sum() override;
    double get_timer_gpu_to_cpu_sum() override;

    double get_timer_knn_elapsed() override;
    double get_timer_lookup_elapsed() override;
    double get_timer_cpu_to_gpu_elapsed() override;
    double get_timer_gpu_to_cpu_elapsed() override;

protected:
    std::unique_ptr<NearestNeighbors> knn;
    std::unique_ptr<Simplex> simplex;
    LUT lut;
    std::vector<float> rhos;
    std::vector<float> buffer;

    double timer_knn_sum = 0;
    double timer_lookup_sum = 0;
    double timer_cpu_to_gpu_sum = 0;
    double timer_gpu_to_cpu_sum = 0;

    double timer_knn_elapsed = 0;
    double timer_lookup_elapsed = 0;
    double timer_cpu_to_gpu_elapsed = 0;
    double timer_gpu_to_cpu_elapsed = 0;
};

#endif
