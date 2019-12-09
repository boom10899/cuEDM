#ifndef __NEAREST_NEIGHBORS_CPU_H__
#define __NEAREST_NEIGHBORS_CPU_H__

#include "lut.h"
#include "nearest_neighbors.h"

class NearestNeighborsCPU : public NearestNeighbors
{
public:
    NearestNeighborsCPU(int tau, int k, bool verbose);

    void compute_lut(LUT &out, const Timeseries &library,
                     const Timeseries &predictee, int E);

protected:
    LUT cache;
};

#endif