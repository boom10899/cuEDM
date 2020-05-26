#include <iostream>

#include "cross_mapping_cpu.h"
#include "stats.h"

// clang-format off
void CrossMappingCPU::run(std::vector<float> &rhos, const Series &library,
                          const std::vector<Series> &targets,
                          const std::vector<uint32_t> &optimal_E, Timer &timer_knn,
                          Timer &timer_lookup)
{
    Timer t1, t2;

    // Compute k-NN lookup tables for library timeseries
    t1.start();
    timer_knn.start();
    for (auto E = 1u; E <= max_E; E++) {
        knn->compute_lut(luts[E - 1], library, library, E);
        luts[E - 1].normalize();
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

double CrossMappingCPU::get_timer_cpu_to_gpu_sum() { return timer_cpu_to_gpu_sum; }
double CrossMappingCPU::get_timer_gpu_to_cpu_sum() { return timer_gpu_to_cpu_sum; }

double CrossMappingCPU::get_timer_cpu_to_gpu_elapsed() { return timer_cpu_to_gpu_elapsed; }
double CrossMappingCPU::get_timer_gpu_to_cpu_elapsed() { return timer_gpu_to_cpu_elapsed; }

// clang-format on
