#include <iostream>
#include "cnpy.h"
#include <cuda.h>
#include <vector>
#include <cmath>
#include <string>

__global__ void PDV_kernel(float* rates_a, float* rates_b, float* results_a, float* results_b, int num_neurons, int num_runs);

int main()
{
    cnpy::NpyArray array_a = cnpy::npy_load("rates_od1.npy"); // will need to pass the precise path from python
    float* a = array_a.data<float>(); // cast into float32
    cnpy::NpyArray array_b = cnpy::npy_load("rates_od2.npy");
    float* b = array_b.data<float>();

    unsigned int num_neurons = &a.size(); // assuming both stimulus were presented to the same ntwrk configuration
    unsigned int num_runs = &a[0].size();

    // here vectors for results
    std::vector<float> res_1(num_runs);
    std::vector<float> res_2(num_runs);

    dim3 numBlocks(num_runs);
    dim3 threadsPerBlock(256);







}

__global__ void PDV_kernel(float* rates_a, float* rates_b, float* results_a, float* results_b, int num_neurons, int num_runs){
    
}
