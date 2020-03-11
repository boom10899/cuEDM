#ifndef __EMBEDDING_DIM_H__
#define __EMBEDDING_DIM_H__

#include <cstdint>

#include "data_frame.h"
#include "timer.h"

class EmbeddingDim
{
public:
    EmbeddingDim(uint32_t max_E, uint32_t tau, uint32_t Tp, bool verbose)
        : max_E(max_E), tau(tau), Tp(Tp), verbose(verbose)
    {
    }
    virtual ~EmbeddingDim() {}

    virtual uint32_t run(const Series &ts, Timer &timer_distance_cal, Timer &timer_lookup) = 0;

protected:
    uint32_t max_E;
    uint32_t tau;
    uint32_t Tp;
    bool verbose;
};

#endif
