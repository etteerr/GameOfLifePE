#include <stdlib.h>
#include "GoLgeneric.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>


__global__ void cuda_kernel(int * src, int * dst, size_t width, size_t height) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; //Linear layout
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx > 0 && idx < width-1 
     && idy > 0 && idy < height-1) 
    { //If we are not a edge
        int acc = 0;
        
        acc = get_rm(src, idx-1, idy+1);
        acc += get_rm(src, idx-1, idy+0);
        acc += get_rm(src, idx-1, idy-1);
        
        acc += get_rm(src, idx, idy+1);
        acc += get_rm(src, idx, idy+0);
        acc += get_rm(src, idx, idy-1);
        
        acc += get_rm(src, idx+1, idy+1);
        acc += get_rm(src, idx+1, idy+0);
        acc += get_rm(src, idx+1, idy-1);
        
        if (acc == 2)
            get_rm(dst, idx, idy) = get_rm(src, idx, idy);
        else if (acc  == 3)
            get_rm(dst, idx, idy) = 1;
        else
            get_rm(dst, idx, idy) = 0;
    }
}