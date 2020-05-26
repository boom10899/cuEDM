#ifndef __CROSS_MAPPING_CPU_H__
#define __CROSS_MAPPING_CPU_H__

#include <memory>

#include "cross_mapping.h"
#include "lut.h"
#include "nearest_neighbors_cpu.h"
#include "simplex_cpu.h"
#include "timer.h"

class CrossMappingCPU : public CrossMapping
{
public:
    CrossMappingCPU(uint32_t max_E, uint32_t tau, uint32_t Tp, bool verbose)
        : CrossMapping(max_E, tau, Tp, verbose),
          knn(new NearestNeighborsCPU(tau, Tp, verbose)),
          simplex(new SimplexCPU(tau, Tp, verbose)), luts(max_E)
    {
    }

    void run(std::vector<float> &rhos, const Series &library,
             const std::vector<Series> &targets,
             const std::vector<uint32_t> &optimal_E, Timer &timer_knn,
             Timer &timer_lookup) override;
    
    double get_timer_cpu_to_gpu_sum() override;
    double get_timer_gpu_to_cpu_sum() override;

    double get_timer_cpu_to_gpu_elapsed() override;
    double get_timer_gpu_to_cpu_elapsed() override;

protected:
    std::unique_ptr<NearestNeighbors> knn;
    std::unique_ptr<Simplex> simplex;
    std::vector<LUT> luts;

    double timer_cpu_to_gpu_sum = 0;
    double timer_gpu_to_cpu_sum = 0;

    double timer_cpu_to_gpu_elapsed = 0;
    double timer_gpu_to_cpu_elapsed = 0;
};

#endif
