#include <stdlib.h>
#include "GoLgeneric.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void cuda_kernel(int * src, int * dst, size_t width, size_t height) {
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int stride = blockDim.x;
    if (idy < height) {
        dim3 p, c, n;

        size_t acc1, acc2, acc3;

        size_t i = threadIdx.x;

        //Initial volley
        size_t idxm1 = (size_t) (i == 0) * width - 1 + (size_t) (i > 0) * (i - 1);
        size_t idxp1 = (size_t) (i + 1 < width) * (i + 1);
        size_t idym1 = (size_t) (idy == 0) * height - 1 + (size_t) (idy > 0) * (i - 1);
        size_t idyp1 = (size_t) (idy + 1 < height) * (idy + 1);

        acc1 = get_rm(src, idxm1, idym1);
        acc2 = get_rm(src, idxm1, idy);
        acc3 = get_rm(src, idxm1, idyp1);

        acc1 += get_rm(src, i, idym1);
        acc2 += get_rm(src, i, idy);
        acc3 += get_rm(src, i, idyp1);

        acc1 += get_rm(src, idxp1, idym1);
        acc2 += get_rm(src, idxp1, idy);
        acc3 += get_rm(src, idxp1, idyp1);

        acc1 += acc2 + acc3;

        get_rm(dst, i, idy) = (acc1 == 2 && get_rm(src, i, idy)) || acc1 == 3;

        //Main chunk
        for (; (i + stride) < width - 1; i += stride) {

            acc1 = get_rm(src, i - 1, idym1);
            acc2 = get_rm(src, i - 1, idy);
            acc3 = get_rm(src, i - 1, idyp1);

            acc1 += get_rm(src, i, idym1);
            acc2 += get_rm(src, i, idy);
            acc3 += get_rm(src, i, idyp1);

            acc1 += get_rm(src, i + 1, idym1);
            acc2 += get_rm(src, i + 1, idy);
            acc3 += get_rm(src, i + 1, idyp1);

            acc1 += acc2 + acc3;

            get_rm(dst, i, idy) = (acc1 == 2 && get_rm(src, i, idy)) || acc1 == 3;
        }

        //Leftovers
        i += stride;

        if (i < width) {
            idxm1 = (size_t) (i == 0) * width - 1 + (size_t) (i > 0) * i - 1;
            idxp1 = (size_t) (i + 1 < width) * (i + 1);

            acc1 = get_rm(src, idxm1, idym1);
            acc2 = get_rm(src, idxm1, idy);
            acc3 = get_rm(src, idxm1, idyp1);

            acc1 += get_rm(src, i, idym1);
            acc2 += get_rm(src, i, idy);
            acc3 += get_rm(src, i, idyp1);

            acc1 += get_rm(src, idxp1, idym1);
            acc2 += get_rm(src, idxp1, idy);
            acc3 += get_rm(src, idxp1, idyp1);
            
            acc1 += acc2 + acc3;

            get_rm(dst, i, idy) = (acc1 == 2 && get_rm(src, i, idy)) || acc1 == 3;
        }
    }
}