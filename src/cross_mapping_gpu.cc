#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <arrayfire.h>

#include "cross_mapping_gpu.h"
#include "stats.h"

CrossMappingGPU::CrossMappingGPU(uint32_t max_E, uint32_t tau, uint32_t Tp,
                                 bool verbose)
    : CrossMapping(max_E, tau, Tp, verbose),
      knn(new NearestNeighborsGPU(tau, Tp, verbose)),
      simplex(new SimplexCPU(tau, Tp, verbose)), luts(max_E)
{
    n_devs = af::getDeviceCount();
}

// clang-format off
void CrossMappingGPU::run(std::vector<float> &rhos, const Series &library,
                          const std::vector<Series> &targets,
                          const std::vector<uint32_t> &optimal_E, Timer &timer_knn,
                          Timer &timer_lookup)
{
    Timer t1, t2;

    Timer timer_cpu_to_gpu;
    Timer timer_gpu_to_cpu;

    timer_cpu_to_gpu_sum = 0;
    timer_gpu_to_cpu_sum = 0;

    // Compute k-NN lookup tables for library timeseries
    t1.start();
    timer_knn.start();
    #pragma omp parallel num_threads(n_devs)
    {
        #ifdef _OPENMP
        uint32_t dev_id = omp_get_thread_num();
        #else
        uint32_t dev_id = 0;
        #endif

        af::setDevice(dev_id);

        // Compute lookup tables for library timeseries
        #pragma omp for schedule(dynamic)
        for (auto E = 1u; E <= max_E; E++) {
            knn->compute_lut(luts[E - 1], library, library, E);
            luts[E - 1].normalize();
        }

        #pragma omp critical 
        {
            timer_cpu_to_gpu_sum += timer_cpu_to_gpu.elapsed();
            timer_gpu_to_cpu_sum += timer_gpu_to_cpu.elapsed();
        }
    }
    timer_knn.stop();
    t1.stop();

    std::vector<float> buffer;
    // Compute Simplex projection from the library to every target
    t2.start();
    timer_lookup.start();
    #pragma omp parallel for private(buffer) schedule(dynamic)
    for (auto i = 0u; i < targets.size(); i++) {
        const auto E = optimal_E[i];

        const auto target = targets[i];
        const auto prediction =
            simplex->predict(buffer, luts[E - 1], target, E);
        const auto shifted_target = simplex->shift_target(target, E);

        rhos[i] = corrcoef(prediction, shifted_target);
    }
    timer_lookup.stop();
    t2.stop();

    if (verbose) {
        std::cout << "k-NN: " << t1.elapsed() << " [ms], Simplex: "
                  << t2.elapsed() << " [ms]" << std::endl;
    }
}

double CrossMappingGPU::get_timer_cpu_to_gpu_sum() { return timer_cpu_to_gpu_sum; }
double CrossMappingGPU::get_timer_gpu_to_cpu_sum() { return timer_gpu_to_cpu_sum; }

double CrossMappingGPU::get_timer_cpu_to_gpu_elapsed() { return timer_cpu_to_gpu_elapsed; }
double CrossMappingGPU::get_timer_gpu_to_cpu_elapsed() { return timer_gpu_to_cpu_elapsed; }

// clang-format on
