#include <iostream>
#include "cnpy.h"
#include <cuda.h>
#include <vector>
#include <cmath>

__global__ void PDV_kernel(float* rates_a, float* rates_b, float* results_a, float* results_b, int num_neurons, int num_runs);

int main()
{
    cnpy::NpyArray array_a = cnpy::npy_load("rates_od1.npy"); // will need to pass the precise path from python
    float* a = array.data_a<float>(); // cast into float32
    cnpy::NpyArray array_b = cnpy::npy_load("rates_od2.npy");
    float* b = array.data_b<float>();

    int num_neurons = data_a.size(); // assuming both stimulus were presented to the same ntwrk configuration
    int num_runs = data_a[0].size();

    // here vector for results
    

    dim3 numBlocks(1000);
    dim3 threadsPerBlock(256);

    PDV_kernel<<<numBlocks, threadsPerBlock>>>(data*, results*, num_neurons, num_runs);

}

__global__ void PDV_kernel(float* rates_a, float* rates_b, float* results_a, float* results_b, int num_neurons, int num_runs){
    
}
