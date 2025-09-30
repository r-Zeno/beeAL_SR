#include <iostream>
#include "cnpy.h"
#include <cuda.h>
#include <vector>

__global__ void PDV_kernel(float* rates, float* results, int num_neurons, int num_runs);

int main()
{
    cnpy::NpyArray array = cnpy::npy_load("neuron_rates.npy"); // will need to pass the precise path from python
    float* data = array.data<float>(); // cast into float32
    int num_neurons = data.size();
    int num_runs = data[0].size();

    // here preallocating array for results

    dim3 numBlocks(1000);
    dim3 threadsPerBlock(256);

    PDV_kernel<<<numBlocks, threadsPerBlock>>>(data*, results*, num_neurons, num_runs);

}

__global__ void PDV_kernel(float* rates, float* results, int num_neurons, in num_runs){
    
}
