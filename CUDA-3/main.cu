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

__global__ void cuda_kernel(int * src, int * dst, size_t width, size_t height);

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
    size_t allocsize = p.width * p.height * sizeof(int);
    int * data = (int*) aligned_alloc(16, allocsize);
    
    if (!data)
        die("Host allocation error");
    
    printf("[%f] Memory allocated (%fMb).\n", toc(), (double)allocsize/1.0e6);
    
    //Pin memory
    mlock((void *)data, p.width * p.height * sizeof(int));
    
    printf("[%f] Memory locked.\n", toc());
    
    //Generate GOL
    struct vector2u size;
    size.x = p.width;
    size.y = p.height;
    if (generateMapInt(data, p.density, p.seed, size))
        exit(3);
    
    printf("[%f] Map generated.\n", toc());
    
    //intial count
    size_t alive = countAliveInt(data, size);
    printf("[%f] Alive: %zu\n", toc(), alive);
    
    //Initializing nvidia (cuda)
    int *cudaSrc, *cudaDst;
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
    
    //Invoke kernel for step times
    dim3 blockDim(16,16);
    dim3 blocks((size.x/blockDim.x)+1,(size.y/blockDim.y)+1);
    int shared = (16+2)*(16+2);
    tic2();
    for(size_t i = 0; i<p.steps/2; i++) {
        cuda_kernel<<<blocks, blockDim, shared*sizeof(int)>>>(cudaSrc, cudaDst, p.width, p.height);
        cuda_kernel<<<blocks, blockDim, shared*sizeof(int)>>>(cudaDst, cudaSrc, p.width, p.height);
    }
    cudaDeviceSynchronize();
    double felaps = toc2();
    checkCuda();
    
    //copy memory back
    tic2();
    cudaMemcpy(data,cudaSrc, allocsize, cudaMemcpyDeviceToHost);
    elaps = toc2();
    checkCuda();
    printf("[%f] Memory copy succesfull, speed=%fGb/s\n", toc(), GET_MEMSPEED(allocsize, elaps)/GBYTE);
    
    //Free
    cudaFree(cudaDst);
    cudaFree(cudaSrc);
    printf("[%f] Memory deallocated.\n", toc());
    
    //Final alive
    alive = countAliveInt(data, size);
    printf("[%f] Alive: %zu\n", toc(), alive);
    
    printf("[%f] Execution succesfull, speed=%fGFLOPS\n", toc(), FLOPS_GOL_INT(p.width, p.height, p.steps, felaps)/GFLOPS);
    printf("[%f] Execution succesfull, GByte/s=%f\n", toc(), MOPS_GOL_INT(4*p.width, p.height, p.steps, felaps)/GBYTE);
    
    cudaProfilerStop();
}