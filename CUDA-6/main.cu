#include <stdio.h> 
#include <sys/mman.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_occupancy.h>
#include <cuda_profiler_api.h>
#include "GoLgeneric.h"
#include "tictoc.h"
#include "defines.h"

__global__ void cuda_kernel(unsigned long long * src, unsigned long long * dst, size_t width, size_t height, unsigned long long mask);

typedef unsigned long long uint64;

void die(char * message) {
    printf("Error: %s\nExiting...\n", message);
    exit(1);
}

void checkCuda() {
    cudaError_t status = cudaPeekAtLastError();
    if (status!=cudaSuccess) {
        fprintf(stderr, "%s: %s\n", cudaGetErrorName(status), cudaGetErrorString(status));
        exit(2);
    }
}

int main(int nargs, char ** args) {
    tic();
    struct inputParameters p;
    
    if (parseInput(nargs, args, &p))
        die("Input error");
    
    printInputParameters(&p);
    
    printf("[%f] initialization done.\n", toc());
    //allocate data
    size_t allocsize = (((p.width)/64) + (size_t)(((p.width)%64)>0)) * (64/8) * p.height;
    uint64 * data = (uint64*) aligned_alloc(32, allocsize);
    
    if (!data)
        die("Host allocation error");
    
    printf("[%f] Memory allocated (%fMb).\n", toc(), (double)allocsize/1.0e6);
    
    //Pin memory
    mlock((void *)data, allocsize);
    
    printf("[%f] Memory locked.\n", toc());
    
    //Generate GOL
    struct vector2u size, tsize;
    size.x = p.width;
    size.y = p.height;
    tsize.x = ((p.width/64) + (size_t)((p.width%64)>0));
    if (generateMapBinary64Little((void*)data, p.density, p.seed, size))
        exit(3);
    
    printf("[%f] Map generated.\n", toc());
    
    //intial count
    size_t alive = countAliveBinary64little((void*)data, size);
    printf("[%f] Alive: %zu\n", toc(), alive);
    
    //Initializing nvidia (cuda)
    uint64 *cudaSrc, *cudaDst;
    cudaMalloc(&cudaSrc, allocsize);
    checkCuda();
    cudaMalloc(&cudaDst, allocsize);
    checkCuda();
    printf("[%f] CUDA initialized and memory allocated.\n", toc());
    
    //copy memory
    tic2();
    cudaMemcpy(cudaSrc, data, allocsize, cudaMemcpyHostToDevice);
    double elaps = toc2();
    checkCuda();
    printf("[%f] Memory copy succesfull, speed=%fGb/s\n", toc(), GET_MEMSPEED(allocsize, elaps)/GBYTE);
    
    //Generate mask
    unsigned long long mask = 0;
    unsigned int extra = (p.width%64);
    for(unsigned int i = 0; i<extra; i++) {
        mask |= (unsigned long long)(1)<<(63-i);
    }
    
    //Invoke kernel for step times
    dim3 blockDim(8,8);
    dim3 blocks((tsize.x/8)+1, (size.y/8)+1);
    tic2();
    for(size_t i = 0; i<p.steps/2; i++) {
        cuda_kernel<<<blocks, blockDim>>>(cudaSrc, cudaDst, p.width, p.height, mask);
        cuda_kernel<<<blocks, blockDim>>>(cudaDst, cudaSrc, p.width, p.height, mask);
    }
    cudaDeviceSynchronize();
    double felaps = toc2();
    checkCuda();
    
    //copy memory back
    tic2();
    cudaMemcpy((void*)data,cudaSrc, allocsize, cudaMemcpyDeviceToHost);
    elaps = toc2();
    checkCuda();
    printf("[%f] Memory copy succesfull, speed=%fGb/s\n", toc(), GET_MEMSPEED(allocsize, elaps)/GBYTE);
    
    //Free
    cudaFree(cudaDst);
    cudaFree(cudaSrc);
    printf("[%f] Memory deallocated.\n", toc());
    
    //Final alive
    alive = countAliveBinary64little(data, size);
    printf("[%f] Alive: %zu\n", toc(), alive);
    
    printf("[%f] Execution succesfull, speed=%fGFLOPS\n", toc(), FLOPS_GOL_INT(p.width, p.height, p.steps, felaps)/GFLOPS);
    printf("[%f] Execution succesfull, GByte/s=%f\n", toc(), MOPS_GOL_INT(4*p.width, p.height, p.steps, felaps)/GBYTE);
    
    cudaProfilerStop();
}