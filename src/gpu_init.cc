#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <arrayfire.h>

#include "gpu_init.h"

void init() {
    Timer timer_gpu_init;
    
    timer_gpu_init.start();

    uint32_t n_devs = af::getDeviceCount();

    #pragma omp parallel num_threads(n_devs)
    {
        #ifdef _OPENMP
        uint32_t dev_id = omp_get_thread_num();
        #else
        uint32_t dev_id = 0;
        #endif

        af::setDevice(dev_id);
    }
    timer_gpu_init.stop();

    std::cout << "GPU init: " << timer_gpu_init.elapsed() << " [ms]" << std::endl;
}   
