#ifndef __CROSS_MAPPING_GPU_H__
#define __CROSS_MAPPING_GPU_H__

#include <memory>

#include "cross_mapping.h"
#include "lut.h"
#include "nearest_neighbors_gpu.h"
#include "simplex_cpu.h"
#include "timer.h"

class CrossMappingGPU : public CrossMapping
{
public:
    CrossMappingGPU(uint32_t max_E, uint32_t tau, uint32_t Tp, bool verbose);

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
    uint32_t n_devs;

    double timer_cpu_to_gpu_sum = 0;
    double timer_gpu_to_cpu_sum = 0;

    double timer_cpu_to_gpu_elapsed = 0;
    double timer_gpu_to_cpu_elapsed = 0;
};

#endif
