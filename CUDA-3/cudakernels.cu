#include <stdlib.h>
#include "GoLgeneric.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define getl(X,Y) local[((X)+1) + (blockDim.x+2) * ((Y)+1)]

__global__ void cuda_kernel(int * src, int * dst, size_t width, size_t height) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    dim3 li(threadIdx.x, threadIdx.y);


    extern __shared__ int local[];

    //Load own
    if (idx < width && idy < height) {
        getl(li.x, li.y) = get_rm(src, idx, idy);

        size_t idxm1 = (size_t) (idx == 0) * width - 1 + (size_t) (idx > 0) * (idx - 1);
        size_t idxp1 = (size_t) (idx + 1 < width) * (idx + 1);
        size_t idym1 = (size_t) (idy == 0) * height - 1 + (size_t) (idy > 0) * (idy - 1);
        size_t idyp1 = (size_t) (idy + 1 < height) * (idy + 1);

        if (li.x == 0) //Left edge
            getl(-1, li.y) = get_rm(src, idxm1, idy);

        if (li.y == 0) //Upper edge
            getl(li.x, -1) = get_rm(src, idx, idym1);

        if (li.x == 0 && li.y == 0) //Upper left corner
            getl(-1, -1) = get_rm(src, idxm1, idym1);

        if (li.x == blockDim.x - 1 || idx == width - 1) // right edge
            getl(li.x + 1, li.y) = get_rm(src, idxp1, idy);

        if (li.y == blockDim.y - 1 || idy == height - 1) //bottom edge
            getl(li.x, li.y + 1) = get_rm(src, idx, idyp1);

        if ((li.y == blockDim.y - 1 || idy == height - 1) && li.x == 0) // lower left corner
            getl(li.x - 1, li.y + 1) = get_rm(src, idxp1, idy);

        if ((li.x == blockDim.x - 1 || idx == width - 1) || idy == 0) //upper right corner
            getl(li.x + 1, li.y - 1) = get_rm(src, idx, idyp1);

        if ((li.y == blockDim.y - 1 || idy == height - 1) && (li.x == blockDim.x - 1 || idx == width - 1)) //lower right corner
            getl(li.x + 1, li.y + 1) = get_rm(src, idxp1, idyp1);
    }
    __syncthreads();

    if (idx < width && idy < height) { //If we are not a edge

        int acc = 0;

        acc += getl(li.x - 1, li.y + 1);
        acc += getl(li.x - 1, li.y + 0);
        acc += getl(li.x - 1, li.y - 1);

        acc += getl(li.x - 0, li.y + 1);
        acc += getl(li.x - 0, li.y + 0);
        acc += getl(li.x - 0, li.y - 1);

        acc += getl(li.x + 1, li.y + 1);
        acc += getl(li.x + 1, li.y + 0);
        acc += getl(li.x + 1, li.y - 1);
        
        //acc = 2 : x * 1 + 0
        //acc = 3 : x * 0 + 1
        //acc = ? : x * 0 + 0
        get_rm(dst, idx, idy) = getl(li.x, li.y) * (int)(acc==2) + (int)(acc==3);
    }
}