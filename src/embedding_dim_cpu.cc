#include <algorithm>

#include "embedding_dim_cpu.h"
#include "stats.h"

uint32_t EmbeddingDimCPU::run(const Series &ts)
{
    // Split input into two halves
    const auto library = ts.slice(0, ts.size() / 2);
    const auto target = ts.slice(ts.size() / 2);

    Timer timer_cpu_to_gpu;
    Timer timer_gpu_to_cpu;

    for (auto E = 1u; E <= max_E; E++) {
        knn->compute_lut(lut, library, target, E, E + 1, timer_cpu_to_gpu, timer_gpu_to_cpu);
        lut.normalize();

        const auto prediction = simplex->predict(buffer, lut, library, E);
        const auto shifted_target = simplex->shift_target(target, E);

        rhos[E - 1] = corrcoef(prediction, shifted_target);
    }

    const auto it = std::max_element(rhos.begin(), rhos.end());
    const auto best_E = it - rhos.begin() + 1;

    return best_E;
}

double EmbeddingDimCPU::get_timer_knn_sum() { return timer_knn_sum; }
double EmbeddingDimCPU::get_timer_lookup_sum() { return timer_lookup_sum; }
double EmbeddingDimCPU::get_timer_cpu_to_gpu_sum() { return timer_cpu_to_gpu_sum; }
double EmbeddingDimCPU::get_timer_gpu_to_cpu_sum() { return timer_gpu_to_cpu_sum; }

double EmbeddingDimCPU::get_timer_knn_elapsed() { return timer_knn_elapsed; }
double EmbeddingDimCPU::get_timer_lookup_elapsed() { return timer_lookup_elapsed; }
double EmbeddingDimCPU::get_timer_cpu_to_gpu_elapsed() { return timer_cpu_to_gpu_elapsed; }
double EmbeddingDimCPU::get_timer_gpu_to_cpu_elapsed() { return timer_gpu_to_cpu_elapsed; }