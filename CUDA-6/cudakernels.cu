#include <stdlib.h>
#include "GoLgeneric.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>

typedef unsigned long long uint64;

__device__ uint64 step64(uint64 rowabove, uint64 rowchunk, uint64 rowbelow) {
    uint64 low, mid, high, c1, c2, a2, a3;
    low = mid = high = 0;

    //above
    c1 = low&rowabove; //Always 0
    low ^= rowabove;
    c2 = mid & c1;
    mid ^= c1;
    high |= c2;

    //below
    c1 = low&rowbelow;
    low ^= rowbelow;
    c2 = mid & c1;
    mid ^= c1;
    high |= c2;

    //upperleft
    c1 = low & (rowabove >> 1);
    low ^= (rowabove >> 1);
    c2 = mid & c1;
    mid ^= c1;
    high |= c2;

    //upperright
    c1 = low & (rowabove << 1);
    low ^= (rowabove << 1);
    c2 = mid & c1;
    mid ^= c1;
    high |= c2;

    //lowerleft
    c1 = low & (rowbelow >> 1);
    low ^= (rowbelow >> 1);
    c2 = mid & c1;
    mid ^= c1;
    high |= c2;

    //lowerright
    c1 = low & (rowbelow << 1);
    low ^= (rowbelow << 1);
    c2 = mid & c1;
    mid ^= c1;
    high |= c2;

    //midleft
    c1 = low & (rowchunk >> 1);
    low ^= (rowchunk >> 1);
    c2 = mid & c1;
    mid ^= c1;
    high |= c2;

    //midright
    c1 = low & (rowchunk << 1);
    low ^= (rowchunk << 1);
    c2 = mid & c1;
    mid ^= c1;
    high |= c2;


    //result
    a2 = (~high) & mid & (~low);
    a3 = (~high) & mid & (low);

    /*
     * if there are 2, nothing happens (if alive, stays alive, if dead stays dead.)
     * if there are 3, new life or nothing
     * Apply border mask
     */
    return ((a2 & rowchunk) | a3)&0b0111111111111111111111111111111111111111111111111111111111111110;
}

#define get64(D,X,Y) D[(X) + (Y)*wd]

__global__ void cuda_kernel(uint64 * src, uint64 * dst, size_t width, size_t height, uint64 mask) {
    int idy = blockIdx.y * blockDim.y + threadIdx.y; //1:1
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //Per 64 cells
    int wd = ((width / 64) + (size_t) ((width % 64) > 0));
    uint64 data1,data2,data3;

    if (idy < height && idx < wd) {
        unsigned int idxm1 = (unsigned int) (idx == 0) * (wd - 1) + (unsigned int) (idx > 0) * (idx - 1);
        unsigned int idxp1 = (unsigned int) (idx + 1 < wd) * (idx + 1);
        unsigned int idym1 = (unsigned int) (idy == 0) * (height - 1) + (unsigned int) (idy > 0) * (idy - 1);
        unsigned int idyp1 = (unsigned int) (idy + 1 < height) * (idy + 1);
        
        unsigned int suml, sumr;
        char bitl, bitr;

 
        data1 = get64(src, idx, idym1);
        data2 = get64(src, idx, idy);
        data3 = get64(src, idx, idyp1);
        
        uint64 res = step64(data1, data2, data3);
        
        sumr = data1 & 0x1 + data2 & 0x1 + data3 & 0x1;
        suml = data1 >> 63 & 0x1 + data2 >> 63 & 0x1 + data3 >> 63 & 0x1;
        bitl = data2 >> 63 & 0x1;
        bitr = data2 & 0x1;
        
        //Calculate bit left
        data1 = get64(src, idxm1, idym1);
        data2 = get64(src, idxm1, idy);
        data3 = get64(src, idxm1, idyp1);

        suml += data1 & 0x1 + data2 & 0x1 + data3 & 0x1;        
        
        //Calculate bit right
        data1 = get64(src, idxp1, idym1);
        data2 = get64(src, idxp1, idy);
        data3 = get64(src, idxp1, idyp1);
        
        sumr += data1 >> 63 & 0x1 + data2 >> 63 & 0x1 + data3 >> 63 & 0x1;
        
        if (idx == wd-1)
            res &= mask;
        
        res |= ((bitl & suml==2) || suml==3) << 63;
        res |= ((bitr & sumr==2) || sumr==3);
        
        get64(dst, idx, idy) = res;
    }
}
