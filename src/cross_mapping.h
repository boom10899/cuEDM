#ifndef __CROSS_MAPPING_H__
#define __CROSS_MAPPING_H__

#include <cstdint>

#include "data_frame.h"
#include "timer.h"

class CrossMapping
{
public:
    CrossMapping(uint32_t max_E, uint32_t tau, uint32_t Tp, bool verbose)
        : max_E(max_E), tau(tau), Tp(Tp), verbose(verbose)
    {
    }
    virtual ~CrossMapping() {}

    virtual void run(std::vector<float> &rhos, const Series &library,
                     const std::vector<Series> &targets,
                     const std::vector<uint32_t> &optimal_E,
                     Timer &timer_knn, Timer &timer_lookup) = 0;

    virtual double get_timer_cpu_to_gpu_sum() = 0;
    virtual double get_timer_gpu_to_cpu_sum() = 0;

    virtual double get_timer_cpu_to_gpu_elapsed() = 0;
    virtual double get_timer_gpu_to_cpu_elapsed() = 0;

protected:
    uint32_t max_E;
    uint32_t tau;
    uint32_t Tp;
    bool verbose;
};

#endif
