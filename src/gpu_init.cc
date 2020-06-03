#include <iostream>
#include <algorithm> 

#ifdef _OPENMP
#include <omp.h>
#endif
#include <arrayfire.h>

#include "gpu_init.h"

void init() {
    Timer timer_gpu_init;

    int data_size = 10;
    std::vector<int> data(data_size);
    std::generate(data.begin(), data.end(), std::rand);

    std::vector<int> data_host1(data_size);
    std::vector<int> data_host2(data_size);
    
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

        af::array data_device1(data.size(), 1, data.data());
        af::array data_device2(data.size(), 1, data.data());

        data_device1.host(data_host1.data());
        data_device2.host(data_host2.data());
    }
    timer_gpu_init.stop();

    std::cout << "GPU init: " << timer_gpu_init.elapsed() << " [ms]" << std::endl;
}   
